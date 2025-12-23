"""
TTT Logger: Unified logging system for SAM-TTT training and inference

功能：
1. 训练日志：loss, delta_norm, grad_norm, lr
2. 推理日志：IoU gate, update count
3. 验证日志：W_init vs cache W comparison
4. TensorBoard 集成（可选）
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch


# ============================================================================
# Logger Configuration
# ============================================================================
@dataclass
class TTTLoggerConfig:
    """日志配置"""
    log_dir: str = "./logs/ttt"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # TensorBoard
    use_tensorboard: bool = False
    tb_log_dir: str = "./runs/ttt"
    
    # 日志频率
    log_every_n_steps: int = 10
    log_first_n_steps: int = 5
    
    # 详细日志
    verbose: bool = True
    log_w_stats: bool = True
    log_grad_stats: bool = True


# ============================================================================
# TTT Logger
# ============================================================================
class TTTLogger:
    """TTT 统一日志系统"""
    
    def __init__(self, config: Optional[TTTLoggerConfig] = None, name: str = "TTT"):
        self.config = config or TTTLoggerConfig()
        self.name = name
        
        # 创建日志目录
        if self.config.log_to_file:
            os.makedirs(self.config.log_dir, exist_ok=True)
        
        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.logger.handlers = []  # 清空已有 handlers
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_format = logging.Formatter(
                '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.config.log_dir, f'ttt_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        
        # TensorBoard writer
        self.tb_writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.config.tb_log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(self.config.tb_log_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available, disabling")
        
        # 统计
        self.step_count = 0
        self.start_time = time.time()
        
    def info(self, msg: str):
        """Info 级别日志"""
        self.logger.info(msg)
        
    def debug(self, msg: str):
        """Debug 级别日志"""
        self.logger.debug(msg)
        
    def warning(self, msg: str):
        """Warning 级别日志"""
        self.logger.warning(msg)
        
    def error(self, msg: str):
        """Error 级别日志"""
        self.logger.error(msg)
        
    def log_step_update(
        self,
        step: int,
        losses: Dict[str, float],
        delta_norms: List[float],
        lr: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """记录 step update 日志"""
        self.step_count += 1
        
        should_log = (
            step <= self.config.log_first_n_steps or
            step % self.config.log_every_n_steps == 0
        )
        
        if should_log:
            msg = f"Step {step}: loss={losses.get('total', 0):.6f}, "
            msg += f"delta_norms={[f'{d:.6f}' for d in delta_norms]}, lr={lr:.4f}"
            
            if extra:
                for k, v in extra.items():
                    if isinstance(v, float):
                        msg += f", {k}={v:.6f}"
                    else:
                        msg += f", {k}={v}"
            
            self.info(msg)
        
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('ttt/total_loss', losses.get('total', 0), step)
            for i, dn in enumerate(delta_norms):
                self.tb_writer.add_scalar(f'ttt/delta_norm_layer{i}', dn, step)
            self.tb_writer.add_scalar('ttt/lr', lr, step)
            
    def log_cache_init(
        self,
        batch_size: int,
        num_layers: int,
        W_init_norms: List[float],
        cache_W_norms: List[float]
    ):
        """记录 cache 初始化日志"""
        self.info(f"Cache initialized: batch_size={batch_size}, layers={num_layers}")
        self.info(f"  W_init norms: {[f'{n:.4f}' for n in W_init_norms]}")
        self.info(f"  Cache W norms: {[f'{n:.4f}' for n in cache_W_norms]}")
        
    def log_iou_gate(
        self,
        step: int,
        pred_iou: float,
        threshold: float,
        should_update: bool
    ):
        """记录 IoU 门控日志"""
        self.info(f"Step {step}: IoU gate - pred_iou={pred_iou:.4f}, "
                  f"thr={threshold:.4f}, update={should_update}")
        
        if self.tb_writer:
            self.tb_writer.add_scalar('ttt/pred_iou', pred_iou, step)
            self.tb_writer.add_scalar('ttt/update_gate', int(should_update), step)
            
    def log_tbptt_detach(self, step: int, layers: List[int]):
        """记录 TBPTT detach 日志"""
        self.info(f"Step {step}: TBPTT detach at layers {layers}")
        
    def log_w_stats(
        self,
        step: int,
        W_init_stats: Dict[str, List[float]],
        cache_W_stats: Dict[str, List[float]]
    ):
        """记录 W 统计信息"""
        if not self.config.log_w_stats:
            return
            
        self.debug(f"Step {step} W stats:")
        self.debug(f"  W_init: norms={W_init_stats.get('norms', [])}, "
                   f"means={W_init_stats.get('means', [])}")
        self.debug(f"  Cache W: norms={cache_W_stats.get('norms', [])}, "
                   f"means={cache_W_stats.get('means', [])}")
                   
    def log_state_dict_check(
        self,
        total_keys: int,
        w_init_keys: List[str],
        expected_layers: int
    ):
        """记录 state_dict 检查日志"""
        self.info(f"State dict check: {total_keys} total keys")
        self.info(f"  W_init keys ({len(w_init_keys)}/{expected_layers}): {w_init_keys}")
        
        if len(w_init_keys) == expected_layers:
            self.info("  ✓ W_init correctly saved in state_dict")
        else:
            self.error(f"  ✗ W_init missing: expected {expected_layers}, got {len(w_init_keys)}")
            
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.info("=" * 60)
        self.info("TTT Training Started")
        self.info("=" * 60)
        for k, v in config.items():
            self.info(f"  {k}: {v}")
        self.info("=" * 60)
        
    def log_training_end(self, total_steps: int, total_time: float):
        """记录训练结束"""
        self.info("=" * 60)
        self.info(f"TTT Training Completed: {total_steps} steps in {total_time:.2f}s")
        self.info(f"  Average: {total_steps/total_time:.2f} steps/sec")
        self.info("=" * 60)
        
    def close(self):
        """关闭 logger"""
        if self.tb_writer:
            self.tb_writer.close()
        for handler in self.logger.handlers:
            handler.close()


# ============================================================================
# Global Logger Instance
# ============================================================================
_global_logger: Optional[TTTLogger] = None


def get_ttt_logger(config: Optional[TTTLoggerConfig] = None) -> TTTLogger:
    """获取全局 TTT logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TTTLogger(config)
    return _global_logger


def log_info(msg: str):
    """便捷函数：info 日志"""
    get_ttt_logger().info(msg)


def log_debug(msg: str):
    """便捷函数：debug 日志"""
    get_ttt_logger().debug(msg)


def log_warning(msg: str):
    """便捷函数：warning 日志"""
    get_ttt_logger().warning(msg)


def log_error(msg: str):
    """便捷函数：error 日志"""
    get_ttt_logger().error(msg)


# ============================================================================
# Self Test
# ============================================================================
if __name__ == "__main__":
    print("Testing TTT Logger...")
    
    config = TTTLoggerConfig(
        log_to_file=False,
        log_to_console=True,
        verbose=True
    )
    
    logger = TTTLogger(config, name="TTT-Test")
    
    # Test basic logging
    logger.info("Test info message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    
    # Test step update logging
    logger.log_step_update(
        step=1,
        losses={'total': 1.5, 'layer0': 0.4},
        delta_norms=[0.001, 0.002, 0.003, 0.004],
        lr=0.01
    )
    
    # Test cache init logging
    logger.log_cache_init(
        batch_size=2,
        num_layers=4,
        W_init_norms=[2.5, 2.6, 2.5, 2.7],
        cache_W_norms=[3.5, 3.7, 3.5, 3.8]
    )
    
    # Test state dict check
    logger.log_state_dict_check(
        total_keys=43,
        w_init_keys=['W_init.0', 'W_init.1', 'W_init.2', 'W_init.3'],
        expected_layers=4
    )
    
    print("\nTTT Logger test completed!")
