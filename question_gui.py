import tkinter as tk
from tkinter import ttk, messagebox
import random
import os

QUESTIONS_FILE = "questions.txt"  # Each line is a question
ANSWERS_FILE = "answers_for_training.txt"  # Each line: question\tanswer

class QuestionGUI:
    def __init__(self, root, nn=None, vectorizer=None):
        self.root = root
        self.nn = nn
        self.vectorizer = vectorizer
        self.root.title("Chatbot for Collecting Training Data")
        self.root.update_idletasks()  # Ensure screen size is correct
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w, h = int(screen_w * 0.8), int(screen_h * 0.8)
        x = (screen_w - w) // 2
        y = int((screen_h - h) * 0.25)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(w, h)  # Enforce minimum size
        self.questions = self.load_questions()
        self.current_question = None
        self.create_widgets()
        self.draw_new_question()

    def load_questions(self):
        if not os.path.exists(QUESTIONS_FILE):
            messagebox.showerror("File Error", f"Questions file '{QUESTIONS_FILE}' not found.")
            self.root.destroy()
            return []
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        if not questions:
            messagebox.showerror("File Error", "No questions found in the file.")
            self.root.destroy()
        return questions

    def create_widgets(self):
        self.root.geometry("900x600")
        self.root.configure(bg="#f5f7fa")
        # Main frame for centering
        main_frame = tk.Frame(self.root, bg="#f5f7fa")
        main_frame.pack(expand=True, fill="both")

        # Chat display area
        self.chat_text = tk.Text(main_frame, wrap="word", font=("Segoe UI", 15), bg="#e3f2fd", fg="#263238", height=20, state="disabled", relief="groove", bd=2)
        self.chat_text.grid(row=0, column=0, columnspan=3, padx=30, pady=(30, 10), sticky="nsew")

        # Entry and send button
        self.answer_var = tk.StringVar()
        self.answer_entry = tk.Entry(main_frame, textvariable=self.answer_var, width=60, font=("Segoe UI", 15), bg="#ffffff", fg="#263238", relief="solid", bd=1)
        self.answer_entry.grid(row=1, column=0, padx=30, pady=10, sticky="ew")
        self.answer_entry.bind('<Return>', lambda event: self.submit_answer())

        self.send_btn = tk.Button(main_frame, text="Send", command=self.submit_answer, bg="#1976d2", fg="white", font=("Segoe UI", 13, "bold"), activebackground="#1565c0", activeforeground="white", relief="flat", padx=10, pady=5, cursor="hand2")
        self.send_btn.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.save_btn = tk.Button(main_frame, text="Save Conversation", command=self.save_conversation, bg="#00bfae", fg="white", font=("Segoe UI", 13, "bold"), activebackground="#008e76", activeforeground="white", relief="flat", padx=10, pady=5, cursor="hand2")
        self.save_btn.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0)
        main_frame.grid_columnconfigure(2, weight=0)

    def draw_new_question(self):
        if not self.questions:
            self.display_message("System", "No questions available.")
            self.send_btn.config(state=tk.DISABLED)
            self.answer_entry.config(state=tk.DISABLED)
            return
        self.current_question = random.choice(self.questions)
        self.display_message("Question", self.current_question)
        self.answer_var.set("")
        self.answer_entry.focus()

    def display_message(self, sender, message):
        self.chat_text.config(state="normal")
        if sender == "Question":
            self.chat_text.insert("end", f"\nQ: {message}\n", "question")
        elif sender == "User":
            self.chat_text.insert("end", f"You: {message}\n", "user")
        else:
            self.chat_text.insert("end", f"{sender}: {message}\n", "system")
        self.chat_text.tag_config("question", foreground="#1565c0", font=("Segoe UI", 13, "bold"))
        self.chat_text.tag_config("user", foreground="#008e76", font=("Segoe UI", 13))
        self.chat_text.tag_config("system", foreground="#b71c1c", font=("Segoe UI", 12, "italic"))
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")

    def submit_answer(self, event=None):
        answer = self.answer_var.get().strip()
        if not answer:
            messagebox.showwarning("Invalid Answer", "Please enter a non-empty answer.")
            return
        self.display_message("User", answer)
        with open(ANSWERS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{self.current_question}\t{answer}\n")
        self.root.after(500, self.draw_new_question)

    def save_conversation(self):
        from tkinter import filedialog
        chat_content = self.chat_text.get("1.0", "end").strip()
        if not chat_content:
            messagebox.showinfo("No Conversation", "There is no conversation to save.")
            return
        file_path = filedialog.asksaveasfilename(title="Save Conversation As", defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(chat_content + "\n" + ("-"*40) + "\n")
        messagebox.showinfo("Saved", f"Conversation saved to {file_path}.")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuestionGUI(root)
    root.mainloop()
