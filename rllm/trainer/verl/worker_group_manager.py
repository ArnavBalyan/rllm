"""Worker Group Manager - Encapsulates worker group initialization and management."""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class WorkerGroupManager(RayPPOTrainer):
    """
    Manages worker groups and provides access to them.
    This decouples worker group management from business logic.
    """
    
    def __init__(self, *args, engine_id, **kwargs):
        self.engine_id = engine_id
        super().__init__(*args, **kwargs)
        self._initialized = False
    
    def init_and_get_worker_groups(self):
        """Initialize workers and return worker group references."""
        if not self._initialized:
            self.init_workers()
            self._initialized = True
        
        return {
            "actor_rollout_wg": self.actor_rollout_wg if hasattr(self, "actor_rollout_wg") else None,
            "critic_wg": self.critic_wg if hasattr(self, "critic_wg") else None,
            "ref_policy_wg": self.ref_policy_wg if hasattr(self, "ref_policy_wg") else None,
            "rm_wg": self.rm_wg if hasattr(self, "rm_wg") else None,
            "async_rollout_manager": self.async_rollout_manager if hasattr(self, "async_rollout_manager") else None,
            # Flags
            "use_critic": self.use_critic,
            "use_reference_policy": self.use_reference_policy,
            "use_rm": self.use_rm,
            "hybrid_engine": self.hybrid_engine,
        }
