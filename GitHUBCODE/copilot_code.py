"""
Create a Rock Paper Scissors game where the player inputs their choice
and plays  against a computer that randomly selects its move, 
with the game showing who won each round.
Add a score counter that tracks player and computer wins, 
and allow the game to continue until the player types “quit”.
"""
from collections import OrderedDict
import random
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

# Game constants
CHOICES = {
    "rock": "🪨 Rock",
    "paper": "📄 Paper", 
    "scissors": "✂️ Scissors"
}

SHORTCUTS = {
    "r": "rock",
    "p": "paper", 
    "s": "scissors"
}

def play_rps(player_choice: str) -> Tuple[str, str]:
    """Play rock paper scissors and return (computer_choice, result)"""
    options = ["rock", "paper", "scissors"]
    computer_choice = random.choice(options)
    
    if player_choice == computer_choice:
        result = "🤝 It's a tie!"
    elif (player_choice == "rock" and computer_choice == "scissors") or \
         (player_choice == "paper" and computer_choice == "rock") or \
         (player_choice == "scissors" and computer_choice == "paper"):
        result = "🎉 You win!"
    else:
        result = "💻 Computer wins!"
    
    return computer_choice, result

def show_welcome():
    """Display welcome message and game rules"""
    print("=" * 50)
    print("🎮 WELCOME TO ROCK PAPER SCISSORS! 🎮")
    print("=" * 50)
    print("Rules:")
    print("🪨 Rock crushes ✂️ Scissors")
    print("📄 Paper covers 🪨 Rock") 
    print("✂️ Scissors cuts 📄 Paper")
    print("\nCommands:")
    print("• Type 'r' or 'rock' for Rock")
    print("• Type 'p' or 'paper' for Paper")
    print("• Type 's' or 'scissors' for Scissors")
    print("• Type 'quit' to exit")
    print("=" * 50)

def display_choices(player_choice: str, computer_choice: str):
    """Display both player and computer choices with formatting"""
    print(f"\n👤 You chose: {CHOICES[player_choice]}")
    print(f"💻 Computer chose: {CHOICES[computer_choice]}")
    print("-" * 30)

def show_stats(stats: Dict[str, int]):
    """Display game statistics"""
    total_games = stats["player"] + stats["computer"] + stats["ties"]
    if total_games == 0:
        return
    
    win_rate = (stats["player"] / total_games) * 100
    print(f"\n📊 FINAL STATISTICS:")
    print(f"Total games played: {total_games}")
    print(f"Your wins: {stats['player']} ({stats['player']/total_games*100:.1f}%)")
    print(f"Computer wins: {stats['computer']} ({stats['computer']/total_games*100:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['ties']/total_games*100:.1f}%)")
    print(f"Your win rate: {win_rate:.1f}%")

def main():
    stats: Dict[str, int] = {"player": 0, "computer": 0, "ties": 0}
    show_welcome()
    
    while True:
        player_input = input("\n🎯 Your choice (r/p/s/quit): ").lower().strip()
        
        # Handle quit
        if player_input == "quit":
            break
            
        # Convert shortcuts to full names
        if player_input in SHORTCUTS:
            player_input = SHORTCUTS[player_input]
        
        # Validate input
        if player_input not in CHOICES:
            print("❌ Invalid input! Please use: r/rock, p/paper, s/scissors, or quit")
            continue
        
        # Play the game
        computer_choice, result = play_rps(player_input)
        
        # Display the round
        display_choices(player_input, computer_choice)
        print(result)
        
        # Update statistics
        if "You win" in result:
            stats["player"] += 1
        elif "Computer wins" in result:
            stats["computer"] += 1
        else:
            stats["ties"] += 1
        
        # Show current score
        print(f"\n🏆 Score: You {stats['player']} - {stats['computer']} Computer (Ties: {stats['ties']})")
        
        # Ask to continue
        continue_game = input("\nPlay again? (y/n): ").lower().strip()
        if continue_game in ['n', 'no']:
            break
    
    # Show final statistics
    show_stats(stats)
    print("\n🎮 Thanks for playing Rock Paper Scissors! 🎮")
class RockPaperScissorsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Rock Paper Scissors Game 🎮")
        self.root.geometry("600x500")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(False, False)
        
        # Game statistics
        self.stats = {"player": 0, "computer": 0, "ties": 0}
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎮 ROCK PAPER SCISSORS 🎮",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        title_label.pack(pady=20)
        
        # Game choices frame
        choices_frame = tk.Frame(self.root, bg="#2c3e50")
        choices_frame.pack(pady=20)
        
        # Player choice buttons
        button_style = {
            "font": ("Arial", 14, "bold"),
            "width": 12,
            "height": 3,
            "relief": "raised",
            "bd": 3
        }
        
        rock_btn = tk.Button(
            choices_frame,
            text="🪨 ROCK",
            bg="#e74c3c",
            fg="white",
            command=lambda: self.play_game("rock"),
            **button_style
        )
        rock_btn.pack(side=tk.LEFT, padx=10)
        
        paper_btn = tk.Button(
            choices_frame,
            text="📄 PAPER",
            bg="#3498db",
            fg="white",
            command=lambda: self.play_game("paper"),
            **button_style
        )
        paper_btn.pack(side=tk.LEFT, padx=10)
        
        scissors_btn = tk.Button(
            choices_frame,
            text="✂️ SCISSORS",
            bg="#2ecc71",
            fg="white",
            command=lambda: self.play_game("scissors"),
            **button_style
        )
        scissors_btn.pack(side=tk.LEFT, padx=10)
        
        # Game result display frame
        result_frame = tk.Frame(self.root, bg="#34495e", relief="sunken", bd=2)
        result_frame.pack(pady=20, padx=20, fill="x")
        
        # Player choice display
        self.player_choice_label = tk.Label(
            result_frame,
            text="👤 Your Choice: Make a selection!",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        )
        self.player_choice_label.pack(pady=5)
        
        # Computer choice display
        self.computer_choice_label = tk.Label(
            result_frame,
            text="💻 Computer Choice: Waiting...",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        )
        self.computer_choice_label.pack(pady=5)
        
        # Result display
        self.result_label = tk.Label(
            result_frame,
            text="🎯 Result: Choose your move!",
            font=("Arial", 14, "bold"),
            bg="#34495e",
            fg="#f39c12"
        )
        self.result_label.pack(pady=10)
        
        # Score display frame
        score_frame = tk.Frame(self.root, bg="#2c3e50")
        score_frame.pack(pady=10)
        
        self.score_label = tk.Label(
            score_frame,
            text="🏆 Score: You 0 - 0 Computer (Ties: 0)",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="#e67e22"
        )
        self.score_label.pack()
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(pady=20)
        
        reset_btn = tk.Button(
            control_frame,
            text="🔄 RESET GAME",
            font=("Arial", 12, "bold"),
            bg="#9b59b6",
            fg="white",
            command=self.reset_game,
            width=15,
            height=2
        )
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = tk.Button(
            control_frame,
            text="❌ QUIT",
            font=("Arial", 12, "bold"),
            bg="#c0392b",
            fg="white",
            command=self.quit_game,
            width=15,
            height=2
        )
        quit_btn.pack(side=tk.LEFT, padx=10)
    
    def play_game(self, player_choice):
        # Use the existing game logic
        computer_choice, result = play_rps(player_choice)
        
        # Update displays
        self.player_choice_label.config(text=f"👤 Your Choice: {CHOICES[player_choice]}")
        self.computer_choice_label.config(text=f"💻 Computer Choice: {CHOICES[computer_choice]}")
        
        # Update result with color coding
        if "You win" in result:
            self.result_label.config(text=f"🎯 {result}", fg="#2ecc71")
            self.stats["player"] += 1
        elif "Computer wins" in result:
            self.result_label.config(text=f"🎯 {result}", fg="#e74c3c")
            self.stats["computer"] += 1
        else:
            self.result_label.config(text=f"🎯 {result}", fg="#f39c12")
            self.stats["ties"] += 1
        
        # Update score display
        self.update_score_display()
    
    def update_score_display(self):
        score_text = f"🏆 Score: You {self.stats['player']} - {self.stats['computer']} Computer (Ties: {self.stats['ties']})"
        self.score_label.config(text=score_text)
    
    def reset_game(self):
        # Reset all statistics
        self.stats = {"player": 0, "computer": 0, "ties": 0}
        
        # Reset displays
        self.player_choice_label.config(text="👤 Your Choice: Make a selection!")
        self.computer_choice_label.config(text="💻 Computer Choice: Waiting...")
        self.result_label.config(text="🎯 Result: Choose your move!", fg="#f39c12")
        self.update_score_display()
        
        messagebox.showinfo("Game Reset", "🎮 Game has been reset! Good luck!")
    
    def quit_game(self):
        total_games = sum(self.stats.values())
        if total_games > 0:
            win_rate = (self.stats["player"] / total_games) * 100
            stats_message = f"""
📊 FINAL STATISTICS:
Total games: {total_games}
Your wins: {self.stats['player']} ({self.stats['player']/total_games*100:.1f}%)
Computer wins: {self.stats['computer']} ({self.stats['computer']/total_games*100:.1f}%)
Ties: {self.stats['ties']} ({self.stats['ties']/total_games*100:.1f}%)
Your win rate: {win_rate:.1f}%

Thanks for playing! 🎮
            """
            messagebox.showinfo("Game Statistics", stats_message)
        
        self.root.quit()

def main_gui():
    """
    Launches the graphical user interface (GUI) version of the Rock-Paper-Scissors game.

    This function initializes the main application window using the Tkinter library,
    creates an instance of the RockPaperScissorsGUI class, and starts the Tkinter event loop.

    Dependencies:
        - The `tk` module from the Tkinter library must be imported.
        - The `RockPaperScissorsGUI` class must be defined elsewhere in the code.

    Raises:
        - Any exceptions raised by the Tkinter library or the RockPaperScissorsGUI class
          during initialization or execution.

    Usage:
        Call this function to start the GUI version of the game.
    """
    """Launch the GUI version of the game"""
    root = tk.Tk()
    game = RockPaperScissorsGUI(root)
    root.mainloop()

if __name__ == "__main__":  
    # Ask user which version they want to play
    print("🎮 Rock Paper Scissors Game 🎮")
    print("Choose your version:")
    print("1. Text-based game")
    print("2. GUI game")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        main_gui()
    else:
        main()