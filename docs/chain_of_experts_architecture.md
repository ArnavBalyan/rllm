# Chain of Experts Architecture

This document provides a comprehensive overview of the Chain of Experts (CoE) architecture and its integration with rLLM components.

## Architecture Overview

The Chain of Experts system extends rLLM's single-agent architecture to support sequential multi-agent workflows where each agent builds upon the output of the previous agent.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Chain of Experts Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MultiAgentExecutionEngine                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │   │
│  │  │ AgentExecEngine │  │ AgentExecEngine │  │ AgentExecEngine │    │   │
│  │  │   [Proposer]    │  │  [Math Expert]  │  │    [Judge]      │    │   │
│  │  │        │        │  │        │        │  │        │        │    │   │
│  │  │        ▼        │  │        ▼        │  │        ▼        │    │   │
│  │  │     Router      │  │     Router      │  │     Router      │    │   │
│  │  │        │        │  │        │        │  │        │        │    │   │
│  │  │        ▼        │  │        ▼        │  │        ▼        │    │   │
│  │  │  vLLM Instance  │  │  vLLM Instance  │  │  vLLM Instance  │    │   │
│  │  │       1         │  │       2         │  │       3         │    │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MultiAgentPPOTrainer                            │   │
│  │              (Integrates with verl training pipeline)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. WorkflowPhase Definition

A `WorkflowPhase` represents a **logical execution phase** in the Chain of Experts:

```python
@dataclass
class WorkflowPhase:
    """
    Represents a logical execution phase in the multi-agent workflow.
    
    For Chain of Experts:
    - Each phase contains exactly ONE agent
    - Phases execute sequentially (Agent A → Agent B → Agent C)
    - Each phase defines WHAT agent executes and HOW it receives context
    
    Note: This is different from a "step" which refers to one turn of conversation/action.
    """
    phase_id: str                   # Unique identifier (e.g., "phase_0", "phase_1")
    agent_ids: List[str]            # List with exactly one agent ID for CoE
    execution_mode: str = "sequential"  # Always "sequential" for Chain of Experts
    description: Optional[str] = None   # Human-readable description
```

**Example Chain of Experts Phases:**
```python
phases = [
    WorkflowPhase(
        phase_id="phase_0",
        agent_ids=["proposer"],
        execution_mode="sequential",
        description="Proposer analyzes problem and suggests initial approach"
    ),
    WorkflowPhase(
        phase_id="phase_1", 
        agent_ids=["math_expert"],
        execution_mode="sequential",
        description="Math Expert receives Proposer output and applies mathematical knowledge"
    ),
    WorkflowPhase(
        phase_id="phase_2",
        agent_ids=["judge"],
        execution_mode="sequential", 
        description="Judge receives Math Expert output and makes final decision"
    )
]
```

**Important Terminology:**
- **Phase**: One agent's execution in the Chain of Experts (new concept)
- **Step**: One conversation turn/action within an agent's execution (preserved from base rLLM)

### 2. Integration with rLLM Router

The Chain of Experts integrates seamlessly with rLLM's routing infrastructure:

```python
class MultiAgentExecutionEngine:
    def _initialize_agent_engines(self):
        """
        Each agent gets its own AgentExecutionEngine which connects to the router.
        The router handles load balancing across vLLM instances.
        """
        for config in self.agent_configs:
            self.agent_engines[config.agent_id] = AgentExecutionEngine(
                engine_name="verl",  # Uses verl backend with router
                tokenizer=self.tokenizer,
                rollout_engine=self.rollout_engine,  # Shared router instance
                agent_class=config.agent_class,
                agent_args=config.agent_args,
                # Agent-specific configuration
                model_path=config.model_path,  # Optional: specialized model
                max_response_length=config.max_response_length,
                max_prompt_length=config.max_prompt_length,
            )
```

**Router Integration Flow:**
1. Each agent has its own `AgentExecutionEngine`
2. Each engine connects to the shared `rollout_engine` (router)
3. Router distributes requests to available vLLM instances based on load
4. Agents can optionally use specialized models via different `model_path`

### 3. vLLM Sharding Strategies

The system supports multiple vLLM sharding strategies:

#### Strategy A: Agent-Specific Models (Recommended)
```python
agent_configs = [
    AgentConfig(
        agent_id="proposer",
        agent_class=ProposerAgent,
        model_path="models/proposer_specialized",  # Creative model for proposals
        temperature=0.8,
    ),
    AgentConfig(
        agent_id="math_expert", 
        agent_class=MathExpertAgent,
        model_path="models/math_specialized",      # Math-optimized model
        temperature=0.3,
    ),
    AgentConfig(
        agent_id="judge",
        agent_class=JudgeAgent,
        model_path="models/reasoning_specialized", # Decision-making model
        temperature=0.2,
    )
]
```

#### Strategy B: Shared Model with Role Conditioning
```python
agent_configs = [
    AgentConfig(
        agent_id="proposer",
        agent_class=ProposerAgent,
        model_path="models/base_model",  # Same base model
        # Role conditioning via system prompts in agent class
    ),
    AgentConfig(
        agent_id="math_expert",
        agent_class=MathExpertAgent, 
        model_path="models/base_model",  # Same base model
        # Math expertise via specialized system prompt
    )
]
```

#### Strategy C: Hybrid Approach
```python
agent_configs = [
    AgentConfig(
        agent_id="proposer",
        model_path="models/base_model",      # General model for initial analysis
    ),
    AgentConfig(
        agent_id="math_expert",
        model_path="models/math_specialist", # Specialized model for math
    ),
    AgentConfig(
        agent_id="judge",
        model_path="models/base_model",      # General model for final decisions
    )
]
```

### 4. Async Engine Integration

Chain of Experts leverages rLLM's async infrastructure at multiple levels:

```python
class MultiAgentExecutionEngine:
    async def execute_multi_agent_trajectory(self, workflow_idx: int, ...):
        """
        Execute a complete Chain of Experts workflow.
        
        Async Integration:
        1. Multiple workflows run in parallel (n_parallel_workflows)
        2. Within each workflow, agents execute sequentially in phases
        3. Each agent uses async execution internally via AgentExecutionEngine
        """
        
        # Sequential execution within the chain (phase by phase)
        for phase in self.workflow_phases:
            phase_output = await self._execute_workflow_phase(...)
            # Pass output to next phase as context
        
    async def _execute_single_agent(self, agent_id: str, ...):
        """
        Execute individual agent using existing async infrastructure.
        """
        engine = self.agent_engines[agent_id]
        
        # Use existing async trajectory execution
        trajectory_result = await engine.run_agent_trajectory_async(
            workflow_idx, application_id, seed, mode, **kwargs
        )
        return trajectory_result
```

**Async Execution Levels:**
- **Workflow Level**: Multiple Chain of Experts workflows run in parallel
- **Phase Level**: Within each workflow, phases execute sequentially
- **Agent Level**: Each agent uses async execution for vLLM interaction
- **Step Level**: Within each agent, conversation steps use existing async patterns

### 5. Context Passing Mechanism

Context flows between agents in the chain through the environment:

```python
# Environment Integration
class MultiAgentEnv(BaseEnv):
    def reset_with_input(self, agent_input: Dict[str, Any]):
        """
        Reset environment with context from previous agents.
        """
        self.multi_agent_context = agent_input
        self._process_multi_agent_input(agent_input)
        
        observation = self._create_observation_with_context()
        return observation, info
    
    def _create_observation_with_context(self):
        """
        Create observation that includes previous agent outputs.
        """
        base_obs = self._create_observation()
        
        # Add collaboration prompt with previous agent outputs
        if self.agent_history:
            base_obs["collaboration_prompt"] = self._format_collaboration_prompt()
        
        return base_obs

# Agent Integration  
class MultiAgentBase(BaseAgent):
    def update_from_env(self, observation, reward, done, info, **kwargs):
        """
        Agent receives context through environment observation.
        """
        if isinstance(observation, dict):
            if "collaboration_prompt" in observation:
                self.multi_agent_context["collaboration"] = observation["collaboration_prompt"]
    
    @property
    def chat_completions(self):
        """
        Convert to chat format including multi-agent context.
        """
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add multi-agent context
        if self.multi_agent_context:
            context_content = self._format_multi_agent_context()
            messages.append({"role": "system", "content": context_content})
        
        return messages
```

**Context Flow:**
1. Agent A produces output → stored in workflow state
2. Agent B starts → `_prepare_agent_input()` retrieves Agent A's output
3. Environment receives context via `reset_with_input()`
4. Environment formats collaboration prompt
5. Agent B receives context through observation
6. Agent B includes context in `chat_completions` for model input

### 6. Training Integration

Chain of Experts integrates with verl's distributed training:

```python
class MultiAgentPPOTrainer(AgentPPOTrainer):
    def fit_multi_agent(self):
        """
        Enhanced training loop that generates Chain of Experts trajectories.
        """
        for batch_dict in self.train_dataloader:
            # Generate Chain of Experts trajectories across phases
            final_gen_batch_output, metrics = self.generate_multi_agent_trajectory(...)
            
            # Continue with standard PPO pipeline
            batch = batch.union(final_gen_batch_output)
            
            # Compute values, advantages, update actor/critic
            # (same as single-agent training on conversation steps)
```

**Training Modes:**
- **"final_agent"**: Train only the final agent using the full chain context
- **"unified"**: Train on combined trajectory from all agents in the chain

**Reward Aggregation:**
- **"final_agent"**: Use reward from the final agent in the chain
- **"average"**: Average rewards across all agents in the chain

**Training Level Distinction:**
- **Phase Level**: Chain of Experts orchestration (new concept)
- **Step Level**: PPO training on conversation turns (preserved from base rLLM)

## Performance Considerations

### 1. Memory Distribution
- Each agent can use separate GPU instances for specialized models
- Router handles load balancing across available vLLM instances
- Memory usage scales linearly with number of agents

### 2. Latency Characteristics
- Sequential execution within chain phases (inherent design)
- Parallel execution across multiple chains
- Context passing adds minimal overhead between phases

### 3. Scalability
```python
# Configuration for scaling
engine = MultiAgentExecutionEngine(
    workflow=chain_workflow,
    n_parallel_workflows=8,        # Run 8 chains in parallel
    trajectory_timeout=300,         # 5-minute timeout per chain
    max_workers=64,                # Thread pool for async operations
)
```

### 4. Resource Requirements
- **GPU Memory**: Depends on model sizes and number of specialized models
- **CPU**: Thread pool for async coordination
- **Network**: Communication between router and vLLM instances

## Getting Started

1. **Define Agent Configs**:
```python
agent_configs = [
    AgentConfig("proposer", ProposerAgent, temperature=0.8),
    AgentConfig("expert", MathExpertAgent, temperature=0.3),
    AgentConfig("judge", JudgeAgent, temperature=0.2),
]
```

2. **Create Workflow**:
```python
workflow = ChainOfExpertsWorkflow(agent_configs)
```

3. **Initialize Engine**:
```python
engine = MultiAgentExecutionEngine(
    workflow=workflow,
    env_class=YourEnvironment,
    engine_name="verl",  # Uses existing router infrastructure
    tokenizer=your_tokenizer,
    rollout_engine=your_rollout_engine,
)
```

4. **Training Integration**:
```python
chain_trainer = create_chain_of_experts_trainer(
    base_trainer=your_base_trainer,
    agent_configs=agent_configs,
    training_mode="final_agent",
)
chain_trainer.fit_multi_agent()
```

## Summary

The Chain of Experts architecture provides a natural extension to rLLM's single-agent system while maintaining compatibility with existing infrastructure components. The key innovation is the clear separation of concerns:

- **Phases**: Sequential agent execution in the Chain of Experts (new multi-agent concept)
- **Steps**: Conversation turns and actions within each agent (preserved rLLM semantics)

This design ensures that existing rLLM components continue to work as expected while enabling powerful multi-agent collaboration patterns. 