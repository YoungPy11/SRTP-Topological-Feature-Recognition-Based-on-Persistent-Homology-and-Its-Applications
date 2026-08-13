"""
============================================================
run_all_pipelines.py — 运行所有 Pipeline 复现
============================================================
SRTP Pipeline 复现项目 · 批量运行脚本

使用方法:
  python run_all_pipelines.py              # 运行所有 pipeline
  python run_all_pipelines.py --paper 1    # 只运行 Paper 1
  python run_all_pipelines.py --paper 2    # 只运行 Paper 2
  ...

对应论文:
  Paper 1 (2505.06583): DNA 结构 PH 分析
  Paper 2 (2601.01450): 喷注标记 (Jet Tagging)
  Paper 3 (2602.14228): TopoGAT 显著特征选择
  Paper 4 (2604.04299): 3DPHDL 设计空间
  Paper 5 (NeurIPS 2023): 自适应过滤学习
============================================================
"""

import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))


def run_paper1():
    """Paper 1: DNA 结构 PH 分析"""
    from pipeline_paper1_dna import main
    main()


def run_paper2():
    """Paper 2: 喷注标记"""
    from pipeline_paper2_jet_tagging import main
    main()


def run_paper3():
    """Paper 3: TopoGAT 显著特征选择"""
    from pipeline_paper3_topogat import main
    main()


def run_paper4():
    """Paper 4: 3DPHDL 设计空间"""
    from pipeline_paper4_3dphdl import main
    main()


def run_paper5():
    """Paper 5: 自适应过滤学习"""
    from pipeline_paper5_filtration_learning import main
    main()


PAPER_RUNNERS = {
    '1': ('Paper 1: DNA 结构 PH 分析', run_paper1),
    '2': ('Paper 2: 喷注标记 (Jet Tagging)', run_paper2),
    '3': ('Paper 3: TopoGAT 显著特征选择', run_paper3),
    '4': ('Paper 4: 3DPHDL 设计空间', run_paper4),
    '5': ('Paper 5: 自适应过滤学习', run_paper5),
}


def main():
    parser = argparse.ArgumentParser(description="运行 SRTP Pipeline 复现")
    parser.add_argument('--paper', type=str, default='all',
                        help="运行指定论文的 pipeline (1-5, 或 all)")
    args = parser.parse_args()

    print("=" * 60)
    print("SRTP Pipeline 复现项目")
    print("基于 TDA 参考资料中的可复现 Pipeline")
    print("=" * 60)

    if args.paper == 'all':
        for paper_id, (name, runner) in PAPER_RUNNERS.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            try:
                runner()
            except Exception as e:
                print(f"  ⚠ {name} 运行出错: {e}")
    else:
        if args.paper in PAPER_RUNNERS:
            name, runner = PAPER_RUNNERS[args.paper]
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            runner()
        else:
            print(f"未知的 Paper 编号: {args.paper}")
            print(f"可用选项: {', '.join(PAPER_RUNNERS.keys())} 或 all")

    print("\n\n" + "=" * 60)
    print("所有 Pipeline 运行完成!")
    print(f"结果保存在: {Path(__file__).parent / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    main()