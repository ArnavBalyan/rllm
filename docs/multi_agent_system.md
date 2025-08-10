# Multi-Agent RL System

This document describes the multi-agent extension for rLLM that enables training and inference with multiple collaborative agents. The system supports various workflow patterns including Chain of Experts, Mixture of Experts, and Multi-Agent Debates.

## Overview

The multi-agent system extends rLLM's single-agent architecture to support:

- **Multiple Agent Types**: Specialized agents for different roles (proposer, critic, judge, etc.)
- **Workflow Orchestration**: Flexible patterns for agent interaction (sequential, parallel, debate)
- **Training Integration**: Seamless integration with existing PPO training pipeline
- **Async Execution**: High-performance async execution for scalable multi-agent inference
- **Generic Architecture**: Easy extension to custom workflow patterns

## Architecture

### Core Components

1. **MultiAgentExecutionEngine**: Orchestrates multi-agent workflows
2. **BaseWorkflow**: Abstract class for defining workflow patterns
3. **AgentConfig**: Configuration for individual agents
4. **MultiAgentEnv**: Environment that handles multi-agent context
5. **MultiAgentPPOTrainer**: Training integration for multi-agent systems

### Workflow Types

#### 1. Chain of Experts (CoE)
Sequential execution where output from Agent A goes to Agent B, etc.

```
Problem → Agent A → Agent B → Agent C → Final Answer
```

**Use Cases:**
- Mathematical problem solving with verification
- Code generation with review and refinement
- Analysis workflows with multiple stages

#### 2. Mixture of Experts (MoE)
Parallel execution with aggregation of expert opinions.

```
         → Expert A →
Problem  → Expert B → Aggregator → Final Answer
         → Expert C →
```

**Use Cases:**
- Complex problems requiring diverse expertise
- Ensemble methods for improved accuracy
- Multi-perspective analysis

#### 3. Multi-Agent Debate
Agents discuss and debate to reach consensus.

```
Problem → Round 1 → Round 2 → Round 3 → Judge → Final Answer
          A ↔ B     A ↔ B     A ↔ B
```

**Use Cases:**
- Decision making with conflicting viewpoints
- Comprehensive analysis of complex issues
- Improving solution quality through discussion

## Quick Start

### 1. Basic Chain of Experts

```python
from rllm.engine.multi_agent_execution_engine import (
    AgentConfig, ChainOfExpertsWorkflow, MultiAgentExecutionEngine
)
from rllm.agents.multi_agent import ProposerAgent, CriticAgent, JudgeAgent

# Configure agents
agent_configs = [
    AgentConfig(
        agent_id="proposer",
        agent_class=ProposerAgent,
        agent_args={"agent_id": "proposer"}
    ),
    AgentConfig(
        agent_id="critic", 
        agent_class=CriticAgent,
        agent_args={"agent_id": "critic"}
    ),
    AgentConfig(
        agent_id="judge",
        agent_class=JudgeAgent,
        agent_args={"agent_id": "judge"}
    ),
]

# Create workflow
workflow = ChainOfExpertsWorkflow(agent_configs)

# Create execution engine
engine = MultiAgentExecutionEngine(
    workflow=workflow,
    env_class=YourEnvironmentClass,
    tokenizer=your_tokenizer,
    # ... other parameters
)

# Execute workflow
tasks = [{"problem": "Your problem here"}]
results = await engine.execute_multi_agent_workflows(tasks)
```

### 2. Training Integration

```python
from rllm.trainer.verl.multi_agent_ppo_trainer import create_chain_of_experts_trainer

# Create base trainer (your existing setup)
base_trainer = AgentPPOTrainer(
    config=your_config,
    tokenizer=your_tokenizer,
    # ... other parameters
)

# Wrap with multi-agent capabilities
multi_agent_trainer = create_chain_of_experts_trainer(
    base_trainer=base_trainer,
    agent_configs=agent_configs,
    training_mode="unified",
    reward_aggregation="final_agent"
)

# Train the multi-agent system
multi_agent_trainer.fit_multi_agent()
```

## Agent Types

### Built-in Agent Classes

- **ProposerAgent**: Proposes initial solutions
- **CriticAgent**: Critiques and refines solutions  
- **JudgeAgent**: Makes final decisions
- **SpecialistAgent**: Domain-specific expertise
- **AggregatorAgent**: Combines multiple inputs
- **DebaterAgent**: Designed for debate scenarios
- **MathExpertAgent**: Specialized for math problems
- **CodeExpertAgent**: Specialized for coding problems
- **ReasoningExpertAgent**: Specialized for logical reasoning

### Creating Custom Agents

```python
from rllm.agents.multi_agent import MultiAgentBase
from rllm.engine.multi_agent_execution_engine import AgentRole

class CustomAgent(MultiAgentBase):
    def __init__(self, **kwargs):
        system_prompt = "Your custom system prompt here"
        super().__init__(role=AgentRole.SPECIALIST, system_prompt=system_prompt, **kwargs)
    
    def _parse_action(self, response: str):
        # Custom action parsing logic
        return response
```

## Environments

### Multi-Agent Environment Base Class

```python
from rllm.environments.multi_agent_env import MultiAgentEnv

class CustomMultiAgentEnv(MultiAgentEnv):
    def _create_observation(self):
        # Create base observation
        return {"problem": self.problem}
    
    def _create_info(self):
        # Create base info
        return {"metadata": self.metadata}
    
    def step(self, action):
        # Environment step logic
        return next_obs, reward, done, info
```

### Built-in Environments

- **MathMultiAgentEnv**: For mathematical problems
- **CodeMultiAgentEnv**: For coding problems  
- **DebateEnv**: For debate scenarios

## Training Modes

### 1. Unified Training
Combines all agent interactions into a single training trajectory.

```python
multi_agent_config = {
    "training_mode": "unified",
    "reward_aggregation": "final_agent"
}
```

### 2. Final Agent Training
Uses only the final agent's trajectory for training.

```python
multi_agent_config = {
    "training_mode": "final_agent",
    "reward_aggregation": "final_agent"
}
```

### 3. Individual Training
Trains each agent separately (requires multiple passes).

```python
multi_agent_config = {
    "training_mode": "individual",
    "reward_aggregation": "average"
}
```

## Reward Aggregation

### Final Agent
Use the reward from the final agent in the workflow.

### Average
Average rewards across all agents.

### Weighted
Weighted average based on agent roles (configurable).

## Custom Workflows

### Creating Custom Workflows

```python
from rllm.engine.multi_agent_execution_engine import BaseWorkflow

class CustomWorkflow(BaseWorkflow):
    def __init__(self, config):
        super().__init__("custom_workflow")
        self.config = config
    
    def define_workflow(self):
        # Define agents
        agents = [...]
        
        # Define execution steps
        steps = [
            WorkflowStep(
                step_id="step1",
                agent_ids=["agent1"],
                execution_mode="sequential"
            ),
            WorkflowStep(
                step_id="step2", 
                agent_ids=["agent2", "agent3"],
                execution_mode="parallel"
            )
        ]
        
        # Define connections
        connections = [
            WorkflowConnection(from_agent="agent1", to_agent="agent2"),
            WorkflowConnection(from_agent="agent1", to_agent="agent3"),
        ]
        
        return agents, steps, connections
    
    def process_step_output(self, step_outputs):
        # Custom processing logic
        return processed_outputs
```

## Performance and Scaling

### GPU Distribution
Each agent type can use separate GPU instances for better resource utilization.

### Async Execution
The system supports fully asynchronous execution for high throughput.

### Memory Management
- Agents share environments when thread-safe
- Context is efficiently passed between agents
- Batched execution for multiple workflows

### Configuration Options

```python
engine = MultiAgentExecutionEngine(
    workflow=workflow,
    n_parallel_workflows=8,  # Concurrent workflows
    trajectory_timeout=300,   # Timeout per workflow
    max_workers=64,          # Thread pool size
    **engine_args
)
```

## Integration with Existing Code

The multi-agent system is designed to integrate seamlessly with existing rLLM code:

1. **Environments**: Extend existing environments or use built-in multi-agent environments
2. **Agents**: Use existing agents or create multi-agent variants
3. **Training**: Wrap existing trainers with multi-agent capabilities
4. **Inference**: Use the same inference patterns with multi-agent execution

## Best Practices

### 1. Agent Design
- Give agents clear, specific roles
- Use appropriate system prompts for each role
- Consider agent interaction patterns

### 2. Workflow Design
- Start simple with 2-3 agents
- Use sequential workflows for dependent tasks
- Use parallel workflows for independent expert opinions

### 3. Training
- Start with "final_agent" training mode
- Use appropriate reward aggregation for your use case
- Monitor multi-agent specific metrics

### 4. Performance
- Use async execution for better throughput
- Configure appropriate timeouts
- Monitor GPU memory usage with multiple agents

## Examples

See `examples/multi_agent_demo.py` for comprehensive examples of:
- Chain of Experts for math problems
- Mixture of Experts for complex analysis
- Multi-Agent Debates for decision making
- Training integration
- Custom workflow creation

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `n_parallel_workflows` or use smaller models
2. **Timeout Errors**: Increase `trajectory_timeout` for complex workflows
3. **Context Issues**: Ensure environments properly handle multi-agent input
4. **Training Convergence**: Try different training modes and reward aggregation

### Debug Options

```python
# Enable debug logging
import logging
logging.getLogger("rllm.engine.multi_agent_execution_engine").setLevel(logging.DEBUG)

# Use visualization
engine.visualize_trajectory(batch, sample_idx=0)
```

## Future Extensions

The multi-agent system is designed for extensibility:

- **New Workflow Patterns**: Hierarchical agents, dynamic workflows
- **Advanced Training**: Multi-agent specific RL algorithms
- **Communication**: Explicit agent-to-agent communication protocols
- **Optimization**: Better resource allocation and load balancing

## References

- [MARTI Framework](https://github.com/TsinghuaC3I/MARTI) - Inspiration for multi-agent patterns
- Original rLLM documentation for base concepts
- verl documentation for training integration 