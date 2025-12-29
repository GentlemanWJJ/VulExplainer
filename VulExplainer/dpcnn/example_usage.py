"""
使用高级分类器的示例代码
展示如何在teacher_main.py中集成新的分类器
"""
import torch
from transformers import RobertaTokenizer, RobertaModel
from model_advanced import AdvancedFusionModel
import argparse

def example_usage():
    """示例：如何使用高级分类器"""
    
    # 1. 准备参数（示例）
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base")
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 2. 加载tokenizer和encoder
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<dis>"])
    codebert = RobertaModel.from_pretrained(args.model_name_or_path)
    codebert.resize_token_embeddings(len(tokenizer))
    
    # 假设有CWE标签映射
    num_classes = 50  # 根据实际CWE数量调整
    
    # 3. 使用不同的分类器类型
    
    # 选项1: MLP分类器（推荐开始使用）
    print("创建MLP分类器模型...")
    model_mlp = AdvancedFusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=num_classes,
        classifier_type='mlp'
    )
    
    # 选项2: 注意力分类器（推荐用于性能提升）
    print("创建注意力分类器模型...")
    model_attention = AdvancedFusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=num_classes,
        classifier_type='attention'
    )
    
    # 选项3: Transformer分类器（推荐用于复杂任务）
    print("创建Transformer分类器模型...")
    model_transformer = AdvancedFusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=num_classes,
        classifier_type='transformer'
    )
    
    # 选项4: 标签感知分类器（推荐用于CWE多分类）
    print("创建标签感知分类器模型...")
    model_label_aware = AdvancedFusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=num_classes,
        classifier_type='label_aware'
    )
    
    # 4. 测试前向传播
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    print("\n测试MLP分类器...")
    model_mlp.eval()
    with torch.no_grad():
        output_mlp = model_mlp(input_ids=input_ids, labels=None)
        print(f"MLP输出形状: {output_mlp.shape}")
    
    print("\n测试注意力分类器...")
    model_attention.eval()
    with torch.no_grad():
        output_attention = model_attention(input_ids=input_ids, labels=None)
        print(f"注意力输出形状: {output_attention.shape}")
    
    print("\n测试Transformer分类器...")
    model_transformer.eval()
    with torch.no_grad():
        output_transformer = model_transformer(input_ids=input_ids, labels=None)
        print(f"Transformer输出形状: {output_transformer.shape}")
    
    print("\n测试标签感知分类器...")
    model_label_aware.eval()
    with torch.no_grad():
        output_label_aware = model_label_aware(input_ids=input_ids, labels=None)
        print(f"标签感知输出形状: {output_label_aware.shape}")
    
    print("\n所有分类器测试完成！")


def modify_teacher_main_example():
    """
    展示如何修改teacher_main.py来使用高级分类器
    
    在teacher_main.py的main()函数中，找到模型初始化部分：
    
    原始代码：
    ```python
    model = FusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=len(cwe_label_map)
    )
    ```
    
    修改为：
    ```python
    from model_advanced import AdvancedFusionModel
    
    # 通过命令行参数选择分类器类型，或直接指定
    classifier_type = getattr(args, 'classifier_type', 'mlp')  # 默认使用MLP
    
    model = AdvancedFusionModel(
        encoder=codebert,
        tokenizer=tokenizer,
        args=args,
        num_class=len(cwe_label_map),
        classifier_type=classifier_type
    )
    ```
    
    然后在argparse中添加参数：
    ```python
    parser.add_argument("--classifier_type", 
                       default="mlp", 
                       type=str,
                       choices=['linear', 'mlp', 'attention', 'residual', 
                               'transformer', 'label_aware', 'contrastive',
                               'hierarchical', 'capsule'],
                       help="分类器类型选择")
    ```
    """
    pass


if __name__ == "__main__":
    example_usage()

