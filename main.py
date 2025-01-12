from imgui_bundle import imgui, immapp
from imgui_bundle import glfw_utils

from Board import Board

import time

TILES_PER_INCH = 5

class GUI:
    def __init__(self):
        self.board = Board(64, 44)
        # create test models at each corner of the board with a radius of 1 inch
        self.board.add_model(2, 0, 0)
        self.board.add_model(0.5, 64, 0)
        self.board.add_model(0.5, 0, 44)
        self.board.add_model(0.5, 64, 44)

        # add terrain to the board def add_terrain(self, terrain: list[list[bool]], startx: int, starty: int)
        # terrain should be a list of lists of booleans where True is a terrain tile and False is not
        # the startx and starty are the coordinates to start drawing the terrain
        # the terrain will be drawn in the same orientation as the list
        # make terrain in the corners of the board offset from the edge by 6 inches and in an corner shape 
        # with a width of 2 inches
        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "top left")
        self.board.add_terrain(terrain, 6, 6)
        self.board.add_terrain(terrain, 6 + 0.2, 6 + 0.2)


        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "top right")
                   
        self.board.add_terrain(terrain, 54, 6)
        self.board.add_terrain(terrain, 54 - 0.2, 6 + 0.2)

        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "bottom left")
        self.board.add_terrain(terrain, 6, 34)
        self.board.add_terrain(terrain, 6 - 0.2, 34 + 0.2)
        

        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "bottom right")
        self.board.add_terrain(terrain, 54, 34)
        self.board.add_terrain(terrain, 54 - 0.2, 34 - 0.2)

        # make a big cross in the center of the board
        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "top left")
        self.board.add_terrain(terrain, 26, 16)
        self.board.add_terrain(terrain, 26 - 0.2, 16 - 0.2)


        terrain = self.board.make_corner_terrain(6 * TILES_PER_INCH, 6 * TILES_PER_INCH, "bottom right")
        self.board.add_terrain(terrain, 26 + 6 - 0.2, 16 + 6 - 0.2)
        self.board.add_terrain(terrain, 26 + 6-0.4, 16 + 6-0.4)


        # make a function that can generate a terrain shape given a width and height and a corner direction
        # the corner direction will be a string that can be "top left", "top right", "bottom left", "bottom right"
        # the width and height are the dimensions of the terrain
        


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