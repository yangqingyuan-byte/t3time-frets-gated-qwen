import torch
from models.T3Time_Learnable_Wavelet_Packet_Gated_Pro_Qwen import TriModalLearnableWaveletPacketGatedProQwen

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    channel = 128
    num_nodes = 7
    seq_len = 96
    pred_len = 96
    wp_level = 2
    
    model = TriModalLearnableWaveletPacketGatedProQwen(
        device=device,
        channel=channel,
        num_nodes=num_nodes,
        seq_len=seq_len,
        pred_len=pred_len,
        wp_level=wp_level
    ).to(device)
    
    # Fake input
    batch_size = 2
    x = torch.randn(batch_size, seq_len, num_nodes).to(device)
    x_mark = torch.randn(batch_size, seq_len, 4).to(device)
    # prompt embedding [B, d_llm, num_nodes, 1] -> [2, 1024, 7, 1]
    emb = torch.randn(batch_size, 1024, num_nodes, 1).to(device)
    
    try:
        output = model(x, x_mark, emb)
        print(f"Success! Output shape: {output.shape}")
        assert output.shape == (batch_size, pred_len, num_nodes)
    except Exception as e:
        print(f"Error during model forward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
