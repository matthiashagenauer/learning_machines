import numpy as np
import os
import pickle
from collections import defaultdict

class ClassicQLearningTester:
    def __init__(self, q_table_path):
        self.q_table = self.load_q_table(q_table_path)
        
    def load_q_table(self, path):
        """Load the Q-table from pickle file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Q-table file not found: {path}")
            
        with open(path, 'rb') as f:
            q_table = pickle.load(f)
            
        # Convert to defaultdict for handling unseen states
        default_q_table = defaultdict(lambda: np.zeros(8))
        default_q_table.update(q_table)
        print(f"✅ Loaded Q-table from {path}")
        return default_q_table
    
    def obs_to_key(self, obs):
        """Convert observation to hashable key (same as in training)"""
        return tuple(obs)
    
    def get_action(self, sensor_values):
        """Get the best action for given sensor values"""
        key = self.obs_to_key(sensor_values)
        return np.argmax(self.q_table[key])
    
    def test_interactive(self):
        """Interactive testing of the Q-table"""
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
                    raise ValueError("Expected exactly 8 integers.")
                
                action = self.get_action(values)
                print(f"🤖 Suggested action: {action}")
                
            except Exception as e:
                print(f"⚠️ Error: {e}\nPlease enter exactly 8 integers.")

if __name__ == "__main__":
    # Update this path to where your Q-table is stored
    Q_TABLE_PATH = "./results/classic_q_table.pkl"
    
    try:
        tester = ClassicQLearningTester(Q_TABLE_PATH)
        tester.test_interactive()
    except Exception as e:
        print(f"❌ Initialization error: {e}")