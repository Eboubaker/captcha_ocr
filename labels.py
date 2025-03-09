import os
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import shutil

class CaptchaLabelingTool:
    def __init__(self, root):
        self.root = root
        self.image_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_index = 0

        # Set up the GUI
        self.root.title("Captcha Labeling Tool")
        self.root.geometry("550x450")
        self.root.resizable(True, True)

        # Directory selection frame
        self.dir_frame = tk.Frame(root, padx=10, pady=10)
        self.dir_frame.pack(fill=tk.X)

        # Input directory selection
        tk.Label(self.dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_dir_var = tk.StringVar()
        self.input_dir_entry = tk.Entry(self.dir_frame, textvariable=self.input_dir_var, width=40)
        self.input_dir_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.dir_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)

        # Output directory selection
        tk.Label(self.dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = tk.Entry(self.dir_frame, textvariable=self.output_dir_var, width=40)
        self.output_dir_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.dir_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # Start button
        tk.Button(self.dir_frame, text="Start Labeling", command=self.start_labeling).grid(row=2, column=0, columnspan=3, pady=10)

        # Main content frame (initially hidden)
        self.content_frame = tk.Frame(root)

        # Frame to display the image
        self.image_frame = tk.Frame(self.content_frame, padx=0, pady=0)  # No padding
        self.image_frame.pack(fill=tk.X, expand=False)

        # File info label (above image)
        self.file_info = tk.Label(self.image_frame, text="", font=("Arial", 14))
        self.file_info.pack(anchor="center")  # Center align

        # Image display
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(anchor="center")  # Center align

        # Frame for input (directly below the image)
        self.input_frame = tk.Frame(self.content_frame)  # No padding
        self.input_frame.pack(fill=tk.X, expand=False)

        # Entry field (without space)
        self.text_entry = tk.Entry(self.input_frame,justify="center", width=60, font=("Arial", 40))
        self.text_entry.pack(fill="x")  # Stretch to fit
        # Buttons frame
        self.button_frame = tk.Frame(self.content_frame)
        self.button_frame.pack(fill=tk.X)

        # Skip button
        self.skip_button = tk.Button(self.button_frame, text="Skip", command=self.skip_image)
        self.skip_button.pack(side=tk.LEFT, padx=5)

        # Submit button
        self.submit_button = tk.Button(self.button_frame, text="Submit", command=self.label_image)
        self.submit_button.pack(side=tk.LEFT, padx=5)

        # Progress label
        self.progress_label = tk.Label(self.content_frame, text="")
        self.progress_label.pack(pady=5)

    def browse_input_dir(self):
        """Browse for input directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_var.set(directory)

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def start_labeling(self):
        """Start the labeling process and remove unused UI elements."""
        self.image_dir = self.input_dir_var.get()
        self.output_dir = self.output_dir_var.get()

        if not self.image_dir or not os.path.isdir(self.image_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return

        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return

        self.image_files = self.get_image_files()

        if not self.image_files:
            messagebox.showerror("Error", "No image files found in the input directory.")
            return

        # Delete directory selection UI elements
        self.dir_frame.destroy()  

        # Show main labeling UI
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self.root.bind('<Return>', lambda event: self.label_image())  # Bind enter key
        self.current_index = 0
        self.load_current_image()
        self.text_entry.focus_set()

    def get_image_files(self):
        """Get all image files from the directory."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        return [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(valid_extensions)]

    def load_current_image(self):
        """Load and display the current image."""
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Completed", "All images have been processed!")
            self.content_frame.pack_forget()
            self.dir_frame.pack(fill=tk.X)
            return

        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, current_file)

        try:
            image = Image.open(image_path)
            display_width = 400
            ratio = display_width / image.width
            display_height = int(image.height * ratio)
            image = image.resize((display_width, display_height), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  

            self.file_info.config(text=f"File: {current_file}")

            self.progress_label.config(text=f"Progress: {self.current_index + 1} of {len(self.image_files)}")

            self.text_entry.delete(0, tk.END)
            self.text_entry.focus_set()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
            self.current_index += 1
            self.load_current_image()

    def label_image(self):
        """Label the current image with the entered text."""
        if self.current_index >= len(self.image_files):
            return

        label_text = self.text_entry.get().strip()

        if not label_text:
            messagebox.showwarning("Warning", "Please enter the captcha text.")
            return

        current_file = self.image_files[self.current_index]
        src_path = os.path.join(self.image_dir, current_file)

        new_filename = f"{label_text}.png"
        dst_path = os.path.join(self.output_dir, new_filename)

        try:
            shutil.move(src_path, dst_path)
            print(f"Moved and labeled: {current_file} -> {new_filename}")

            self.current_index += 1
            self.load_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Error saving labeled file: {e}")

    def skip_image(self):
        """Skip the current image."""
        self.current_index += 1
        self.load_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = CaptchaLabelingTool(root)
    root.mainloop()
