#!/usr/bin/env python3
"""
启动Nightingale AI后端服务器的脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # 检查是否在正确的目录
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("错误: 找不到backend目录")
        print("请确保在项目根目录运行此脚本")
        sys.exit(1)
    
    # 检查requirements.txt
    requirements_file = backend_dir / "requirements.txt"
    if not requirements_file.exists():
        print("错误: 找不到requirements.txt文件")
        sys.exit(1)
    
    print("🚀 启动Nightingale AI后端服务器...")
    print("📍 后端地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        # 切换到backend目录并启动服务器
        os.chdir("backend")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✅ 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        print("\n请检查:")
        print("1. 是否安装了所有依赖: pip install -r backend/requirements.txt")
        print("2. Python版本是否兼容 (推荐 3.8+)")
        sys.exit(1)

if __name__ == "__main__":
    main()
