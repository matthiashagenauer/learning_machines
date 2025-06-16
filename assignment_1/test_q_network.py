import torch
import numpy as np
import os

# Define the same QNetwork architecture used in training
class QNetwork(torch.nn.Module):
    def __init__(self, num_hidden=128):
        super(QNetwork, self).__init__()
        self.l1 = torch.nn.Linear(8, num_hidden)
        self.l2 = torch.nn.Linear(num_hidden, 8)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        return self.l2(x)

def load_model(model_path):
    model = QNetwork()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to eval mode for inference
        print(f"✅ Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    return model

def test_model(model):
    print("\n📥 Enter 8 sensor values (integers from 0 to 100+), separated by spaces.")
    print("  → These simulate proximity sensor values.")
    print("  → You can enter `exit` to quit.\n")

    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            values = list(map(int, user_input.split()))
            if len(values) != 8:
                raise ValueError("Expected 8 integers.")
            input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(input_tensor)
                best_action = torch.argmax(q_values).item()
            print(f"🎯 Q-values: {q_values.numpy().squeeze()}")
            print(f"🤖 Suggested action: {best_action}")
        except Exception as e:
            print(f"⚠️ Error: {e}\nPlease enter exactly 8 integers.")

if __name__ == "__main__":
    model_path = ".\\results\\q_network_params.pth"  # Update if needed
    model = load_model(model_path)
    test_model(model)
