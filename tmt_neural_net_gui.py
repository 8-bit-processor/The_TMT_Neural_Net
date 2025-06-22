import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from tmt_neural_net import NeuralNetwork
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tmt_text_preprocessor import TextPreprocessor
from question_gui import QuestionGUI

class NeuralNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Trainer")
        # Make window nearly full screen and responsive
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w, h = int(screen_w * 0.80), int(screen_h * 0.80)
        x, y = (screen_w - w) // 2, (screen_h - h) // 2-60
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(int(screen_w * 0.7), int(screen_h * 0.7))
        # Modern theme and background
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(bg="#f5f7fa")
        style.configure("TLabelFrame", background="#f5f7fa", borderwidth=0, relief="flat")
        style.configure("TFrame", background="#f5f7fa")
        style.configure("TLabel", background="#f5f7fa", font=("Segoe UI", 14))
        # Lighter color for Predict button
        style.configure("TButton", font=("Segoe UI", 14, "bold"), padding=8, relief="flat", background="#e3f2fd", foreground="#222")
        style.map("TButton",
                  background=[('active', '#bbdefb'), ('!active', '#e3f2fd')],
                  foreground=[('active', '#222'), ('!active', '#222')])
        style.configure("TEntry", font=("Segoe UI", 14))
        style.configure("TCombobox", font=("Segoe UI", 14))
        self.vectorizer = None  # Store vectorizer for saving/loading
        self.create_widgets()

    def create_widgets(self):
        # Frame for parameters
        param_frame = ttk.LabelFrame(self.root, text="Training Parameters", padding=12)
        param_frame.grid(row=0, column=0, padx=16, pady=6, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=2)
        self.root.grid_rowconfigure(2, weight=6)
        self.root.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(1, weight=1)
        param_frame.grid_columnconfigure(2, weight=1)
        param_frame.grid_columnconfigure(3, weight=1)
        param_frame.grid_columnconfigure(4, weight=1)
        param_frame.grid_columnconfigure(5, weight=1)

        # Row 0: Learning rate, Iterations
        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.E, padx=8, pady=8)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(param_frame, textvariable=self.lr_var, width=10, style="TEntry").grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(param_frame, text="Training Iterations:").grid(row=0, column=2, sticky=tk.E, padx=8, pady=8)
        self.iter_var = tk.IntVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.iter_var, width=10, style="TEntry").grid(row=0, column=3, padx=8, pady=8)

        # Row 1: Hidden layers, Output size
        ttk.Label(param_frame, text="Hidden Layer 1 Neurons:").grid(row=1, column=0, sticky=tk.E, padx=8, pady=8)
        self.h1_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.h1_var, width=10, style="TEntry").grid(row=1, column=1, padx=8, pady=8)
        ttk.Label(param_frame, text="Hidden Layer 2 Neurons:").grid(row=1, column=2, sticky=tk.E, padx=8, pady=8)
        self.h2_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.h2_var, width=10, style="TEntry").grid(row=1, column=3, padx=8, pady=8)
        ttk.Label(param_frame, text="Output Size:").grid(row=1, column=4, sticky=tk.E, padx=8, pady=8)
        self.out_var = tk.IntVar(value=1)
        ttk.Entry(param_frame, textvariable=self.out_var, width=10, style="TEntry").grid(row=1, column=5, padx=8, pady=8)

        # Row 2: Buttons
        btn_frame = tk.Frame(param_frame, bg="#f5f7fa")
        btn_frame.grid(row=2, column=0, columnspan=6, pady=16)
        # Launch Question GUI button (leftmost)
        self.question_gui_btn = ttk.Button(btn_frame, text="Launch Question GUI", command=self.launch_question_gui, style="TButton")
        self.question_gui_btn.pack(side=tk.LEFT, padx=12)
        # Batch Train button (next)
        self.batch_train_btn = ttk.Button(btn_frame, text="Batch Train from Answers", command=self.load_and_batch_train, style="TButton")
        self.batch_train_btn.pack(side=tk.LEFT, padx=12)
        # Pre-process Corpus button (next)
        self.preprocess_btn = ttk.Button(btn_frame, text="Pre-process Corpus into Training Dataset", command=self.preprocess_corpus_to_dataset, style="TButton")
        self.preprocess_btn.pack(side=tk.LEFT, padx=12)
        # Train
        self.train_btn = ttk.Button(btn_frame, text="Train", command=self.train_network, style="TButton")
        self.train_btn.pack(side=tk.LEFT, padx=12)
        # Save/Load Model (rightmost, with file dialog)
        self.save_btn = ttk.Button(btn_frame, text="Save Model", command=self.save_model_as, style="TButton")
        self.save_btn.pack(side=tk.LEFT, padx=12)
        self.load_btn = ttk.Button(btn_frame, text="Load Model", command=self.load_model_as, style="TButton")
        self.load_btn.pack(side=tk.LEFT, padx=12)

        # Matplotlib figure for error plot
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('#f5f7fa')
        self.ax.set_facecolor('#e3f2fd')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=16, pady=4, sticky="nsew")

        # Chat interface (minimal padding, balanced)
        chat_frame = ttk.LabelFrame(self.root, text="Chat", padding=8)
        chat_frame.grid(row=2, column=0, padx=8, pady=4, sticky="nsew")
        chat_frame.grid_columnconfigure(0, weight=8)
        chat_frame.grid_columnconfigure(1, weight=1)
        chat_frame.grid_columnconfigure(2, weight=0)
        chat_frame.grid_rowconfigure(0, weight=8)
        chat_frame.grid_rowconfigure(1, weight=1)

        # Chat history (scrollable, smaller size for chat look)
        self.chat_text = tk.Text(chat_frame, height=12, width=48, font=("Segoe UI", 14), bg="#fff", fg="#000", wrap="word", state="disabled", relief="groove", bd=2, padx=10, pady=10, highlightthickness=1, highlightbackground="#90caf9")
        self.chat_text.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        chat_scroll = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        chat_scroll.grid(row=0, column=2, sticky="ns", pady=(0, 6))
        self.chat_text['yscrollcommand'] = chat_scroll.set

        # User input, predict button only (smaller input for chat look)
        self.user_input = tk.Entry(chat_frame, font=("Segoe UI", 14), bg="#fff", relief="groove", bd=2)
        self.user_input.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=2, ipady=4)
        self.user_input.bind('<Return>', lambda event: self.predict_chat())
        btn_frame = tk.Frame(chat_frame, bg="#f5f7fa")
        btn_frame.grid(row=1, column=1, sticky="nsew", pady=2)
        btn_frame.grid_columnconfigure(0, weight=1)
        predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict_chat, style="TButton")
        predict_btn.grid(row=0, column=0, sticky="nsew")
        chat_frame.grid_columnconfigure(0, weight=8)
        chat_frame.grid_columnconfigure(1, weight=1)

    def send_message(self, event=None):
        user_msg = self.user_input.get().strip()
        if not user_msg:
            return
        self.append_chat("You", user_msg)
        self.user_input.delete(0, tk.END)
        # No GPT or dummy response, do nothing

    def predict_chat(self):
        user_msg = self.user_input.get().strip()
        if not user_msg:
            return
        # Show user input in chatbox as a chat message
        self.append_chat("You", user_msg)
        self.user_input.delete(0, tk.END)
        try:
            if not hasattr(self, 'nn') or self.nn is None:
                self.append_chat("Prediction", "(Please train or load the neural network first.)")
                return
            # If sequence model (output size > 1), predict next sentence
            if hasattr(self, 'vectorizer') and self.vectorizer is not None and hasattr(self, 'nn') and self.nn is not None and getattr(self.nn, 'output_size', 1) > 1:
                X_pred = self.vectorizer.transform([user_msg]).toarray()
                y_pred = self.nn.predict(X_pred)
                # Find closest sentence in vectorizer vocabulary
                vocab = self.vectorizer.get_feature_names_out()
                # Reconstruct sentence from predicted vector (pick top N words)
                top_indices = np.argsort(y_pred[0])[::-1][:10]
                top_words = [vocab[i] for i in top_indices if y_pred[0][i] > 0]
                predicted_sentence = ' '.join(top_words) if top_words else "(No prediction)"
                self.append_chat("Prediction", predicted_sentence)
                return
            # Use Q/A pairs from answers_for_training.txt for prediction
            qa_pairs = []
            try:
                with open("answers_for_training.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        if '\t' in line:
                            q, a = line.strip().split('\t', 1)
                            qa_pairs.append((q.strip(), a.strip()))
            except Exception:
                self.append_chat("Prediction", "(Could not read answers_for_training.txt)")
                return
            if not qa_pairs:
                self.append_chat("Prediction", "(No Q/A pairs in answers_for_training.txt)")
                return
            questions = [q for q, a in qa_pairs]
            answers = [a for q, a in qa_pairs]
            # Fit vectorizer on questions
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(questions)
            X_pred = vectorizer.transform([user_msg])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(X_pred, X_train).flatten()
            best_idx = int(np.argmax(similarities))
            best_answer = answers[best_idx] if similarities[best_idx] > 0 else "(No close match in training data)"
            self.append_chat("Prediction", best_answer)
        except Exception as e:
            self.append_chat("Prediction", f"(Prediction error: {e})")

    def append_chat(self, sender, message):
        self.chat_text.config(state="normal")
        if sender == "You":
            self.chat_text.insert(tk.END, f"You: {message}\n")
        elif sender == "Prediction":
            self.chat_text.insert(tk.END, f"Prediction: {message}\n")
        else:
            self.chat_text.insert(tk.END, f"{sender}: {message}\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state="disabled")

    def get_training_texts(self):
        # Try to load from answers_for_training.txt and conversation_log.txt
        texts = set()
        try:
            with open("answers_for_training.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if '\t' in line:
                        texts.add(line.split('\t', 1)[1].strip())
        except Exception:
            pass
        try:
            with open("conversation_log.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not set(line.strip()) == {'-'}]
                for i in range(len(lines)):
                    if lines[i].startswith("You: "):
                        texts.add(lines[i][4:].strip())
        except Exception:
            pass
        return sorted(texts)

    def get_questions_list(self):
        try:
            with open("questions.txt", "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]
            return questions
        except Exception:
            return []

    def train_network(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(title="Select Q/A File", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        try:
            lr = self.lr_var.get()
            iterations = self.iter_var.get()
            h1 = self.h1_var.get()
            h2 = self.h2_var.get()
            output_size = self.out_var.get()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return
        # Training data: load from selected file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                messagebox.showwarning("No Data", "No answers found in the selected file.")
                return
            # Use Q/A pairs if tab-separated, else treat as sequence pairs
            X_data, y_data = [], []
            for line in lines:
                if "\t" in line:
                    q, a = line.split("\t", 1)
                    X_data.append(q.strip())
                    y_data.append(a.strip())
            if not X_data or not y_data:
                messagebox.showwarning("No Data", "No valid Q/A pairs found in the selected file.")
                return
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(X_data).toarray()
            # For a simple regression/classification, use y as text and vectorize
            y_vec = self.vectorizer.transform(y_data).toarray() if output_size > 1 else np.zeros(X.shape[0])
            self.nn = NeuralNetwork(input_size=X.shape[1], hidden_size1=h1, hidden_size2=h2, output_size=output_size, learning_rate=lr)
            error_list = self.nn.train(X, y_vec, training_iterations=iterations)
            self.ax.clear()
            self.ax.plot(error_list)
            self.ax.set_xlabel("Iterations (x100)")
            self.ax.set_ylabel("Cumulative Error")
            self.ax.set_title("Training Error Over Time (TF-IDF)")
            self.canvas.draw()
            # Show performance in chatbox
            final_error = error_list[-1] if error_list else None
            self.append_chat("System", f"Training complete. Final error: {final_error:.4f}" if final_error is not None else "Training complete.")
            messagebox.showinfo("Training Complete", "Neural network training finished!")
        except FileNotFoundError:
            messagebox.showerror("File Error", f"{file_path} not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        return

    def load_and_batch_train(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(title="Select Conversation Log File", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Split into Q/A pairs, ignore separator lines
            lines = [line.strip() for line in content.splitlines() if line.strip() and not set(line.strip()) == {'-'}]
            questions = []
            answers = []
            for i in range(len(lines)):
                if lines[i].startswith("Q: ") and i+1 < len(lines) and lines[i+1].startswith("You: "):
                    questions.append(lines[i][3:].strip())
                    answers.append(lines[i+1][4:].strip())
            if not answers:
                messagebox.showwarning("No Data", "No Q/A pairs found in the selected file.")
                return
            # Vectorize answers (or questions+answers) using TF-IDF
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(answers).toarray()
            y = np.zeros(X.shape[0])
            self.nn = NeuralNetwork(input_size=X.shape[1], learning_rate=self.lr_var.get(), hidden_size1=self.h1_var.get(), hidden_size2=self.h2_var.get(), output_size=self.out_var.get())
            error_list = self.nn.train(X, y, training_iterations=self.iter_var.get())
            self.ax.clear()
            self.ax.plot(error_list)
            self.ax.set_xlabel("Iterations (x100)")
            self.ax.set_ylabel("Cumulative Error")
            self.ax.set_title("Batch Training Error (TF-IDF Conversation Log)")
            self.canvas.draw()
            messagebox.showinfo("Batch Training Complete", "Batch training on conversation log complete!")
        except FileNotFoundError:
            messagebox.showerror("File Error", f"{file_path} not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def preprocess_corpus_to_dataset(self):
        """
        Open a file dialog to select a text corpus, preprocess it, and save normalized sentence pairs to corpus_dataset.txt.
        """
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(title="Select Text Corpus", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            preprocessor = TextPreprocessor()
            filtered = preprocessor.filter_sentences(preprocessor.split_sentences(text))
            X_sent, y_sent = preprocessor.prepare_sequence_pairs(filtered)
            # Remove any lines where either side is empty after normalization and avoid duplicates
            seen_pairs = set()
            with open("corpus_dataset.txt", "w", encoding="utf-8") as out_f:
                for x, y_ in zip(X_sent, y_sent):
                    norm_x = preprocessor.normalize_sentence(x)
                    norm_y = preprocessor.normalize_sentence(y_)
                    if norm_x and norm_y:
                        pair = (norm_x, norm_y)
                        if pair not in seen_pairs:
                            out_f.write(f"{norm_x}\t{norm_y}\n")
                            seen_pairs.add(pair)
            messagebox.showinfo("Pre-processing Complete", "Corpus has been pre-processed and saved to corpus_dataset.txt.")
        except Exception as e:
            messagebox.showerror("Pre-processing Error", str(e))

    def launch_question_gui(self):
        # Launch the Question GUI popup in a new Toplevel window
        popup_win = tk.Toplevel(self.root)
        popup = QuestionGUI(popup_win, nn=self.nn if hasattr(self, 'nn') else None, vectorizer=self.vectorizer if hasattr(self, 'vectorizer') else None)
        popup_win.transient(self.root)
        popup_win.grab_set()
        popup_win.focus()

    def save_model_as(self):
        from tkinter import filedialog
        if not hasattr(self, 'nn') or self.nn is None or self.vectorizer is None:
            messagebox.showwarning("Not Trained", "Please train the network before saving.")
            return
        file_path = filedialog.asksaveasfilename(title="Save Model As", defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return
        try:
            with open(file_path, "wb") as f:
                pickle.dump({'model': self.nn, 'vectorizer': self.vectorizer}, f)
            messagebox.showinfo("Model Saved", f"Trained model and vectorizer saved to {file_path}.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_model_as(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(title="Load Model", filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.nn = data['model']
            self.vectorizer = data['vectorizer']
            messagebox.showinfo("Model Loaded", f"Trained model and vectorizer loaded from {file_path}.")
        except FileNotFoundError:
            messagebox.showerror("File Error", f"{file_path} not found.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def predict_input(self):
        if not hasattr(self, 'nn') or self.nn is None:
            messagebox.showwarning("Not Trained", "Please train the network first.")
            return
        try:
            input_str = self.input_var.get().strip()
            if not input_str:
                messagebox.showwarning("No Input", "Please select or enter a question.")
                return
            all_texts = self.get_training_texts()
            if not all_texts:
                self.prediction_label.config(text="Prediction: (No training data)")
                return
            if self.vectorizer is not None:
                vectorizer = self.vectorizer
            else:
                vectorizer = TfidfVectorizer()
                vectorizer.fit(all_texts)
            X_train = vectorizer.transform(all_texts)
            X_pred = vectorizer.transform([input_str])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(X_pred, X_train).flatten()
            best_idx = int(np.argmax(similarities))
            best_text = all_texts[best_idx] if similarities[best_idx] > 0 else "(No close match in training data)"
            self.prediction_label.config(text=f"Prediction: {best_text}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetGUI(root)
    root.mainloop()
