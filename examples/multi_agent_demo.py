#!/usr/bin/env python3
"""
Multi-Agent RL Training Demo

This script demonstrates how to use the multi-agent extension for rLLM.
It shows examples of:
1. Chain of Experts (CoE) - Sequential agent execution
2. Mixture of Experts (MoE) - Parallel agent execution with aggregation  
3. Multi-Agent Debate - Agents discuss and collaborate
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
    MixtureOfExpertsWorkflow,
    DebateWorkflow,
    MultiAgentExecutionEngine,
)
from rllm.agents.multi_agent import (
    ProposerAgent,
    CriticAgent,
    JudgeAgent,
    MathExpertAgent,
    CodeExpertAgent,
    ReasoningExpertAgent,
    AggregatorAgent,
    DebaterAgent,
)
from rllm.environments.multi_agent_env import MathMultiAgentEnv, DebateEnv
from rllm.trainer.verl.multi_agent_ppo_trainer import (
    create_chain_of_experts_trainer,
    create_mixture_of_experts_trainer,
    create_debate_trainer,
)


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


# Example 1: Chain of Experts for Math Problems
def demo_chain_of_experts():
    """Demonstrate Chain of Experts workflow for math problems"""
    print("\n=== CHAIN OF EXPERTS DEMO ===")
    
    # Configure agents
    agent_configs = [
        AgentConfig(
            agent_id="proposer",
            agent_class=ProposerAgent,
            role=AgentRole.PROPOSER,
            agent_args={"agent_id": "proposer"}
        ),
        AgentConfig(
            agent_id="math_expert",
            agent_class=MathExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "math_expert", "specialty": "Mathematics"}
        ),
        AgentConfig(
            agent_id="critic",
            agent_class=CriticAgent,
            role=AgentRole.CRITIC,
            agent_args={"agent_id": "critic"}
        ),
        AgentConfig(
            agent_id="judge",
            agent_class=JudgeAgent,
            role=AgentRole.JUDGE,
            agent_args={"agent_id": "judge"}
        ),
    ]
    
    # Create workflow
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
    
    print(f"Running Chain of Experts on task: {tasks[0]['problem']}")
    print(f"Expected solution: {tasks[0]['solution']}")
    
    # Run async execution (in real scenario)
    async def run_coe():
        results = await engine.execute_multi_agent_workflows(tasks)
        return results
    
    # For demo, we'll simulate the execution
    print("Chain of Experts workflow would execute:")
    print("1. Proposer: Analyzes the problem and proposes initial approach")
    print("2. Math Expert: Applies mathematical knowledge to solve")
    print("3. Critic: Reviews the solution for errors")  
    print("4. Judge: Makes final decision on the answer")
    
    return workflow, engine


# Example 2: Mixture of Experts for Complex Problems
def demo_mixture_of_experts():
    """Demonstrate Mixture of Experts workflow"""
    print("\n=== MIXTURE OF EXPERTS DEMO ===")
    
    # Configure expert agents
    expert_configs = [
        AgentConfig(
            agent_id="math_expert",
            agent_class=MathExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "math_expert", "specialty": "Mathematics"}
        ),
        AgentConfig(
            agent_id="code_expert",
            agent_class=CodeExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "code_expert", "specialty": "Programming"}
        ),
        AgentConfig(
            agent_id="reasoning_expert",
            agent_class=ReasoningExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "reasoning_expert", "specialty": "Logic"}
        ),
    ]
    
    # Configure aggregator
    aggregator_config = AgentConfig(
        agent_id="aggregator",
        agent_class=AggregatorAgent,
        role=AgentRole.AGGREGATOR,
        agent_args={"agent_id": "aggregator"}
    )
    
    # Create workflow
    workflow = MixtureOfExpertsWorkflow(expert_configs, aggregator_config)
    
    # Create execution engine
    tokenizer = create_demo_tokenizer()
    engine = MultiAgentExecutionEngine(
        workflow=workflow,
        env_class=MathMultiAgentEnv,
        env_args={},
        engine_name="openai",
        tokenizer=tokenizer,
        n_parallel_workflows=1,
    )
    
    print("Mixture of Experts workflow would execute:")
    print("1. All experts work in parallel on the same problem")
    print("2. Math Expert: Focuses on mathematical aspects")
    print("3. Code Expert: Considers algorithmic approaches")
    print("4. Reasoning Expert: Analyzes logical structure")
    print("5. Aggregator: Combines all expert opinions into final answer")
    
    return workflow, engine


# Example 3: Multi-Agent Debate
def demo_multi_agent_debate():
    """Demonstrate Multi-Agent Debate workflow"""
    print("\n=== MULTI-AGENT DEBATE DEMO ===")
    
    # Configure debater agents
    debater_configs = [
        AgentConfig(
            agent_id="debater_for",
            agent_class=DebaterAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "debater_for", "position": "For"}
        ),
        AgentConfig(
            agent_id="debater_against",
            agent_class=DebaterAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "debater_against", "position": "Against"}
        ),
        AgentConfig(
            agent_id="moderator",
            agent_class=ReasoningExpertAgent,
            role=AgentRole.SPECIALIST,
            agent_args={"agent_id": "moderator", "specialty": "Moderation"}
        ),
    ]
    
    # Configure judge
    judge_config = AgentConfig(
        agent_id="judge",
        agent_class=JudgeAgent,
        role=AgentRole.JUDGE,
        agent_args={"agent_id": "judge"}
    )
    
    # Create workflow
    workflow = DebateWorkflow(debater_configs, judge_config, max_rounds=3)
    
    # Create execution engine
    tokenizer = create_demo_tokenizer()
    engine = MultiAgentExecutionEngine(
        workflow=workflow,
        env_class=DebateEnv,
        env_args={},
        engine_name="openai",
        tokenizer=tokenizer,
        n_parallel_workflows=1,
    )
    
    print("Multi-Agent Debate workflow would execute:")
    print("1. Round 1: Each debater presents initial position")
    print("2. Round 2: Debaters respond to each other's arguments")
    print("3. Round 3: Final arguments and rebuttals")
    print("4. Judge: Evaluates all arguments and reaches conclusion")
    
    return workflow, engine


# Training Integration Demo
def demo_training_integration():
    """Show how to integrate multi-agent workflows with training"""
    print("\n=== TRAINING INTEGRATION DEMO ===")
    
    print("To integrate with training, you would:")
    print("1. Create your base AgentPPOTrainer")
    print("2. Use factory functions to wrap it with multi-agent capabilities")
    print("3. Call fit_multi_agent() instead of fit_agent()")
    
    print("""
Example code:

# Create base trainer (using your existing setup)
base_trainer = AgentPPOTrainer(
    config=your_config,
    tokenizer=your_tokenizer,
    # ... other parameters
)

# Create multi-agent workflow
agent_configs = [
    AgentConfig(agent_id="expert1", agent_class=MathExpertAgent, ...),
    AgentConfig(agent_id="expert2", agent_class=CodeExpertAgent, ...),
]

# Wrap with multi-agent capabilities
multi_agent_trainer = create_chain_of_experts_trainer(
    base_trainer=base_trainer,
    agent_configs=agent_configs,
    training_mode="unified",  # or "final_agent", "individual"
    reward_aggregation="final_agent"  # or "average", "weighted"
)

# Train the multi-agent system
multi_agent_trainer.fit_multi_agent()
""")


# Advanced Workflows Demo
def demo_custom_workflow():
    """Show how to create custom workflows"""
    print("\n=== CUSTOM WORKFLOW DEMO ===")
    
    print("You can create custom workflows by extending BaseWorkflow:")
    print("""
from rllm.engine.multi_agent_execution_engine import BaseWorkflow

class CustomWorkflow(BaseWorkflow):
    def __init__(self, config):
        super().__init__("custom_workflow")
        self.config = config
    
    def define_workflow(self):
        # Define your agents, steps, and connections
        agents = [...]
        steps = [...]
        connections = [...]
        return agents, steps, connections
    
    def process_step_output(self, step_outputs):
        # Custom processing logic
        return processed_outputs
""")


# Performance and Scaling Demo
def demo_scaling():
    """Discuss performance and scaling considerations"""
    print("\n=== SCALING AND PERFORMANCE ===")
    
    print("Key considerations for scaling multi-agent RL:")
    print("1. GPU Memory: Each agent may need separate GPU instances")
    print("2. Communication: Agents need to share context efficiently")
    print("3. Load Balancing: Distribute work across available resources")
    print("4. Async Execution: Use async patterns for better throughput")
    print("5. Caching: Cache agent responses for repeated interactions")
    
    print("\nConfiguration options:")
    print("- n_parallel_workflows: Number of concurrent workflows")
    print("- training_mode: unified/final_agent/individual")
    print("- reward_aggregation: How to combine rewards from multiple agents")
    print("- async_engine: Enable asynchronous execution")


def main():
    """Run all demos"""
    print("Multi-Agent RL System Demo")
    print("=" * 50)
    
    try:
        # Run each demo
        demo_chain_of_experts()
        demo_mixture_of_experts()
        demo_multi_agent_debate()
        demo_training_integration()
        demo_custom_workflow()
        demo_scaling()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("1. Set up your model and tokenizer")
        print("2. Configure your environment and reward functions")
        print("3. Create agent configurations for your use case")
        print("4. Choose appropriate workflow type")
        print("5. Run training with fit_multi_agent()")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("This is expected in a demo environment without full setup")


if __name__ == "__main__":
    main() 