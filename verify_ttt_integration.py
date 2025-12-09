import torch
import sys
import os

# 将当前目录添加到路径中，以便我们可以导入 sam2
sys.path.append(os.getcwd())

print("正在检查模块导入...")
try:
    from sam2.modeling.sam_ttt.DWT import extract_high_frequency
    from sam2.modeling.sam_ttt.mix_embedding import ME
    from sam2.modeling.sam_ttt.Route_Fuse import routefuse
    from sam2.modeling.sam_ttt.ttt import TTTLinear
    print("✅ 核心模块导入成功。")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 发生未知错误: {e}")
    sys.exit(1)

print("正在检查模块初始化...")
try:
    # 测试 DWT
    dwt = extract_high_frequency()
    print("✅ DWT 初始化成功。")

    # 测试 ME (模拟 SAM2 的 hidden_dim=256)
    # 在 sam2_base.py 中我们使用了: ME(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim)
    me = ME(in_channels=512, out_channels=256)
    print("✅ ME (Mix Embedding) 初始化成功。")

    # 测试 RouteFuse
    rf = routefuse(256, 256)
    print("✅ RouteFuse 初始化成功。")
    
    print("\n🎉 所有新模块均已通过基础验证！代码结构正常。")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
