import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

UNIT = 100  # pixels
HEIGHT = 9  # grid height
WIDTH = 9  # grid width

class MarkovProcessDisplay(tk.Tk):
    def __init__(self, markov_process, start_state):
        super(MarkovProcessDisplay, self).__init__()
        self.title('Markov Process (Random Walk): Click on Next Step or press Space to move the agent')
        self.geometry(f'{WIDTH * UNIT}x{HEIGHT * UNIT + 50}')
        self.markov_process = markov_process
        self.start_state = start_state
        self.current_state = start_state
        self.step_count = 0
        self.running = False
        self.visit_counts = np.zeros(markov_process.num_states)
        self.canvas = self._build_canvas()
        self.rect = None
        self.step_text = None
        self._draw_grid()
        self._draw_agent()
        self._create_buttons()
        self._display_step_count()
        self.bind('<space>', self.space_key)
        self.prob_window = None

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        canvas.pack()
        return canvas

    def _draw_grid(self):
        for c in range(0, WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, HEIGHT * UNIT, fill='black')
        for r in range(0, HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, WIDTH * UNIT, r, fill='black')

    def _draw_agent(self):
        x, y = self._state_to_xy(self.current_state)
        self.rect = self.canvas.create_oval(
            x + 25, y + 25, x + 75, y + 75, fill='red')

    def _state_to_xy(self, state):
        col = state[0] #% WIDTH
        row = state[1] #% WIDTH
        return col * UNIT, row * UNIT

    def _move_agent(self, next_state):
        self.canvas.delete(self.rect)
        x, y = self._state_to_xy(next_state)
        self.rect = self.canvas.create_oval(
            x + 25, y + 25, x + 75, y + 75, fill='red')
        self.current_state = next_state

    def _create_buttons(self):
        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X)
        
        next_step_button = tk.Button(button_frame, text="Next Step", command=self.next_step)
        next_step_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.toggle_button = tk.Button(button_frame, text="Start Continuous", command=self.toggle_continuous)
        self.toggle_button.pack(side=tk.LEFT, padx=10, pady=10)

        prob_button = tk.Button(button_frame, text="Show Probabilities", command=self.show_probabilities)
        prob_button.pack(side=tk.LEFT, padx=10, pady=10)

    def next_step(self, event=None):
        next_state = self.markov_process.next_state(self.current_state)
        self._move_agent(next_state)
        self.step_count += 1
        self.visit_counts[next_state[0]] += 1
        self._display_step_count()

    def toggle_continuous(self):
        self.running = not self.running
        if self.running:
            self.toggle_button.config(text="Stop Continuous")
            self.run_continuous()
        else:
            self.toggle_button.config(text="Start Continuous")

    def run_continuous(self):
        if self.running:
            self.next_step()
            self.after(100, self.run_continuous)
    
    def space_key(self, event):
        self.next_step()

    def _display_step_count(self):
        if self.step_text:
            self.canvas.delete(self.step_text)
        self.step_text = self.canvas.create_text(
            10, 10, anchor='nw', text=f"Step: {self.step_count}", font=('Helvetica', 16), fill='black')

    def show_probabilities(self):
        if self.prob_window is None or not tk.Toplevel.winfo_exists(self.prob_window):
            self.prob_window = tk.Toplevel(self)
            self.prob_window.title("State Visit Probabilities")
            self._update_probabilities()

    def _update_probabilities(self):
        if self.prob_window is not None:
            fig, ax = plt.subplots()
            states = np.arange(self.markov_process.num_states)
            probabilities = self.visit_counts / self.visit_counts.sum()
            ax.bar(states, probabilities)
            ax.set_xlabel('States')
            ax.set_ylabel('Probability')
            ax.set_title('State Visit Probabilities')

            canvas = FigureCanvasTkAgg(fig, master=self.prob_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class MarkovProcess:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        self.num_states = transition_matrix.shape[0]
    
    def next_state(self, current_state):
        x=np.random.choice(self.num_states, p=self.transition_matrix[current_state[0]])
        y=np.random.choice(self.num_states, p=self.transition_matrix[current_state[1]])
        return x,y

if __name__ == "__main__":
    pc=0.1  # probability of not moving
    pp=0.8  # probability of moving proximally
    pcc=0.3 # probability of not moving at the boundary
    transition_matrix = np.array([
        [pcc, 1-pcc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1-pc-pp,   pc,  pp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1-pc-pp,   pc,  pp, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1-pc-pp,  pc,  pp, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, (1-pc)/2,  pc,  (1-pc)/2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, pp,  pc, 1-pc-pp, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, pp, pc, 1-pc-pp, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pp, pc, 1-pc-pp],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1-pcc, pcc]
    ])
    markov_process = MarkovProcess(transition_matrix)

    start_state = [0,0]

    app = MarkovProcessDisplay(markov_process, start_state)
    app.mainloop()
