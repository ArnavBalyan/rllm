#!/usr/bin/env python3
"""
Chain of Experts RL Training Demo

This script demonstrates how to use the Chain of Experts (CoE) extension for rLLM.
Chain of Experts enables sequential agent execution where:
- Agent A processes the initial problem
- Agent B receives Agent A's output as context and refines it  
- Agent C receives Agent B's output and provides final solution
- Each agent can use specialized models via separate vLLM instances
"""

import asyncio
import os
import sys
from typing import List

# Add rllm to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from rllm.engine.multi_agent_execution_engine import (
    AgentConfig,
    AgentRole,
    ChainOfExpertsWorkflow,
    MultiAgentExecutionEngine,
)
from rllm.agents.multi_agent import (
    ProposerAgent,
    CriticAgent,
    JudgeAgent,
    MathExpertAgent,
    CodeExpertAgent,
    ReasoningExpertAgent,
)
from rllm.environments.multi_agent_env import MathMultiAgentEnv
from rllm.trainer.verl.multi_agent_ppo_trainer import create_chain_of_experts_trainer


def create_demo_tokenizer():
    """Create a demo tokenizer (you would use your actual model tokenizer)"""
    try:
        return AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        # Fallback for demo purposes
        print("Warning: Using mock tokenizer for demo")
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                
            def encode(self, text, add_special_tokens=False):
                # Simple word-based tokenization for demo
                return [hash(word) % 1000 for word in text.split()]
                
            def decode(self, tokens):
                return f"decoded_text_{len(tokens)}_tokens"
        
        return MockTokenizer()


def demo_chain_of_experts():
    """Demonstrate Chain of Experts workflow for math problems"""
    print("\n=== CHAIN OF EXPERTS DEMO ===")
    print("Sequential execution: Proposer → Math Expert → Critic → Judge")
    
    # Configure agents in chain order
    agent_configs = [
        AgentConfig(
            agent_id="proposer",
            agent_class=ProposerAgent,
            role=AgentRole.PROPOSER,
            agent_args={"agent_id": "proposer"},
            model_path="models/proposer_model",  # Optional: specialized model
            temperature=0.8,  # Creative for initial proposals
        ),
        AgentConfig(
            agent_id="math_expert",
            agent_class=MathExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "math_expert", "specialty": "Mathematics"},
            model_path="models/math_expert_model",  # Optional: math-specialized model
            temperature=0.3,  # Lower temp for precise calculations
        ),
        AgentConfig(
            agent_id="critic",
            agent_class=CriticAgent,
            role=AgentRole.CRITIC,
            agent_args={"agent_id": "critic"},
            temperature=0.5,  # Balanced for analysis
        ),
        AgentConfig(
            agent_id="judge",
            agent_class=JudgeAgent,
            role=AgentRole.JUDGE,
            agent_args={"agent_id": "judge"},
            temperature=0.2,  # Low temp for final decisions
        ),
    ]
    
    # Create Chain of Experts workflow
    workflow = ChainOfExpertsWorkflow(agent_configs)
    
    # Create execution engine
    tokenizer = create_demo_tokenizer()
    engine = MultiAgentExecutionEngine(
        workflow=workflow,
        env_class=MathMultiAgentEnv,
        env_args={},
        engine_name="openai",  # or "verl" for training
        tokenizer=tokenizer,
        n_parallel_workflows=1,
    )
    
    # Demo task
    tasks = [
        {
            "problem": "Solve for x: 2x + 5 = 17",
            "solution": "x = 6",
        }
    ]
    
    print(f"\nTask: {tasks[0]['problem']}")
    print(f"Expected solution: {tasks[0]['solution']}")
    
    print("\nChain of Experts execution flow:")
    print("Step 0: Proposer analyzes the problem and proposes initial approach")
    print("Step 1: Math Expert receives Proposer's output and applies mathematical knowledge")
    print("Step 2: Critic receives Math Expert's output and reviews for errors")  
    print("Step 3: Judge receives Critic's output and makes final decision")
    
    # Actual execution would be:
    # results = await engine.execute_multi_agent_workflows(tasks)
    
    return workflow, engine


def demo_chain_architecture():
    """Explain the Chain of Experts architecture"""
    print("\n=== CHAIN OF EXPERTS ARCHITECTURE ===")
    
    print("""
Chain of Experts Architecture Integration with rLLM:

┌─────────────────────────────────────────────────────────────────┐
│                    Chain of Experts Workflow                    │
├─────────────────────────────────────────────────────────────────┤
│  Step 0: [Proposer Agent]     ──→ Router ──→ vLLM Instance 1   │
│     ↓ (output as context)                                      │
│  Step 1: [Math Expert Agent]  ──→ Router ──→ vLLM Instance 2   │
│     ↓ (output as context)                                      │
│  Step 2: [Critic Agent]       ──→ Router ──→ vLLM Instance 3   │
│     ↓ (output as context)                                      │
│  Step 3: [Judge Agent]        ──→ Router ──→ vLLM Instance 4   │
│                                                                 │
│  MultiAgentPPOTrainer integrates with verl for distributed     │
│  training across all agent instances                           │
└─────────────────────────────────────────────────────────────────┘

Key Architecture Points:

1. WorkflowStep Definition:
   - Each step contains exactly one agent for Chain of Experts
   - Steps execute sequentially, not in parallel
   - Each step can have different execution modes (currently "sequential")

2. vLLM Sharding Options:
   - Option A: Each agent gets its own specialized model/vLLM instance
   - Option B: All agents share the same base model with role conditioning
   - Router handles load balancing and GPU distribution

3. Context Passing:
   - MultiAgentEnv.reset_with_input() passes previous agent output
   - Environment formats collaboration prompts automatically
   - Agent receives context through observation in chat_completions

4. Async Integration:
   - Uses existing rLLM async infrastructure
   - Each agent has its own AgentExecutionEngine
   - Workflows can run in parallel while agents within workflow run sequentially

5. Training Integration:
   - "final_agent" mode: train only the last agent in chain
   - "unified" mode: train on combined trajectory from all agents
   - Reward aggregation: use final agent reward or average across chain
""")


def demo_training_integration():
    """Show how to integrate Chain of Experts with training"""
    print("\n=== TRAINING INTEGRATION DEMO ===")
    
    print("To integrate Chain of Experts with training:")
    print("1. Create your base AgentPPOTrainer (existing setup)")
    print("2. Define agent configs in chain order")
    print("3. Use create_chain_of_experts_trainer() factory function")
    print("4. Call fit_multi_agent() instead of fit_agent()")
    
    print("""
Example integration code:

# Step 1: Create base trainer (your existing setup)
base_trainer = AgentPPOTrainer(
    config=your_config,
    tokenizer=your_tokenizer,
    role_worker_mapping=your_role_mapping,
    resource_pool_manager=your_resource_manager,
    # ... other parameters
)

# Step 2: Define Chain of Experts agents
agent_configs = [
    AgentConfig(
        agent_id="proposer", 
        agent_class=ProposerAgent,
        model_path="models/proposer_model",  # Optional: specialized model
        temperature=0.8
    ),
    AgentConfig(
        agent_id="expert", 
        agent_class=MathExpertAgent,
        model_path="models/math_model",  # Optional: math-specialized model  
        temperature=0.3
    ),
    AgentConfig(
        agent_id="judge", 
        agent_class=JudgeAgent,
        temperature=0.2
    ),
]

# Step 3: Wrap with Chain of Experts capabilities
chain_trainer = create_chain_of_experts_trainer(
    base_trainer=base_trainer,
    agent_configs=agent_configs,
    training_mode="final_agent",  # or "unified"
    reward_aggregation="final_agent"  # or "average"
)

# Step 4: Train the Chain of Experts system
chain_trainer.fit_multi_agent()

Training Modes:
- "final_agent": Only train the final agent (Judge) using Chain context
- "unified": Train on combined trajectory from all agents in chain

Reward Aggregation:
- "final_agent": Use reward from final agent in chain
- "average": Average rewards across all agents in chain
""")


def demo_custom_agents():
    """Show how to create custom agents for Chain of Experts"""
    print("\n=== CUSTOM AGENTS DEMO ===")
    
    print("Creating custom agents for specific Chain of Experts roles:")
    print("""
from rllm.agents.multi_agent import MultiAgentBase
from rllm.engine.multi_agent_execution_engine import AgentRole

class CustomReviewerAgent(MultiAgentBase):
    def __init__(self, **kwargs):
        system_prompt = '''You are a Reviewer Agent in a Chain of Experts.
        Your role is to:
        1. Review the work from the previous agent in the chain
        2. Identify any issues or improvements needed
        3. Provide specific feedback and corrections
        4. Prepare refined output for the next agent
        
        Consider the previous agent's work carefully and build upon it.'''
        
        super().__init__(
            role=AgentRole.CRITIC, 
            system_prompt=system_prompt, 
            **kwargs
        )
    
    def _parse_action(self, response: str):
        # Custom parsing logic if needed
        return response

# Usage in Chain of Experts
agent_configs = [
    AgentConfig("analyzer", AnalyzerAgent),
    AgentConfig("reviewer", CustomReviewerAgent),  # Your custom agent
    AgentConfig("finalizer", FinalizerAgent),
]
""")


def demo_scaling_considerations():
    """Discuss scaling and performance considerations"""
    print("\n=== SCALING AND PERFORMANCE ===")
    
    print("Key considerations for scaling Chain of Experts:")
    print("""
1. GPU Memory Distribution:
   - Each agent can use separate GPU instances
   - Specialized models for different roles (math, code, reasoning)
   - Router handles load balancing across vLLM instances

2. Sequential vs Parallel Execution:
   - Agents within chain execute sequentially (by design)
   - Multiple chains can execute in parallel (n_parallel_workflows)
   - Async execution within each agent for better throughput

3. Model Specialization Options:
   - Option A: Different models per agent (Proposer, Expert, Critic, Judge)
   - Option B: Same base model with different system prompts
   - Option C: Hybrid approach with specialized models for key roles

4. Context Management:
   - Efficient passing of context between chain steps
   - MultiAgentEnv handles context formatting automatically
   - Configurable max_prompt_length per agent

5. Training Efficiency:
   - "final_agent" mode reduces training overhead
   - "unified" mode captures full chain behavior
   - Gradient accumulation across chain steps

Configuration example:
engine = MultiAgentExecutionEngine(
    workflow=chain_workflow,
    n_parallel_workflows=8,    # Parallel chains
    trajectory_timeout=300,     # Per-chain timeout
    max_workers=64,            # Thread pool size
)
""")


def main():
    """Run Chain of Experts demo"""
    print("Chain of Experts RL System Demo")
    print("=" * 50)
    
    try:
        # Run each demo section
        demo_chain_of_experts()
        demo_chain_architecture()
        demo_training_integration()
        demo_custom_agents()
        demo_scaling_considerations()
        
        print("\n" + "=" * 50)
        print("Chain of Experts demo completed successfully!")
        print("\nNext steps:")
        print("1. Set up your models and tokenizer")
        print("2. Configure your environment and reward functions")
        print("3. Create agent configurations in chain order")
        print("4. Choose training mode (final_agent vs unified)")
        print("5. Run training with fit_multi_agent()")
        print("\nKey benefits of Chain of Experts:")
        print("- Sequential refinement of solutions")
        print("- Specialized agents for different roles")
        print("- Efficient context passing between agents")
        print("- Flexible training modes and reward aggregation")
        print("- Integration with existing rLLM infrastructure")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("This is expected in a demo environment without full setup")


if __name__ == "__main__":
    main() 