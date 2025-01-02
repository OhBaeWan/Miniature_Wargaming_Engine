from imgui_bundle import imgui, immapp
from imgui_bundle import glfw_utils

from Board import Board

import time

class GUI:
    def __init__(self):
        self.board = Board(64, 44)
        # create test models at each corner of the board with a radius of 1 inch
        self.board.add_model(2, 0, 0)
        self.board.add_model(0.5, 64, 0)
        self.board.add_model(0.5, 0, 44)
        self.board.add_model(0.5, 64, 44)

    def draw(self):
        # start a group for the left side of the screen
        imgui.begin_group()
        self.board.update()
        imgui.text("")
        # end the group
        imgui.end_group()
        imgui.same_line()
        # start a group for the right side of the screen
        imgui.begin_group()
        imgui.text("Right Side")
        imgui.end_group()
        
    
    def run(self):
        immapp.run(
            gui_function=self.draw,  # The Gui function to run
            window_title="Minature Tabletop Engine",  # the window title
            window_size_auto=True,  # Auto size the application window given its widgets
            # Uncomment the next line to restore window position and size from previous run
            window_restore_previous_geometry=True,
            fps_idle=60,  # The maximum frame rate when the window is not active
        )
        


if __name__ == "__main__":
    gui = GUI()
    gui.run()