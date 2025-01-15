from imgui_bundle import imgui, immapp
from imgui_bundle import glfw_utils
import glfw 
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.breadth_first import BreadthFirstFinder
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.ida_star import IDAStarFinder
import time
import math

import numpy as np
from typing import Any
from numpy.typing import NDArray
from enum import Enum
import cv2  # type: ignore
import math
from imgui_bundle import immvision
import threading
from multiprocessing import Process, Queue



TILES_PER_INCH = 5

immvision.use_rgb_color_order()


class Board:
    def __init__(self, sizex: float, sizey: float):
        self.sizex = sizex
        self.sizey = sizey

    def get_neighbors(self, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((self.board[y][x - 1], 1))
        if x < self.sizex * TILES_PER_INCH - 1:
            neighbors.append((self.board[y][x + 1], 1))
        if y > 0:
            neighbors.append((self.board[y - 1][x], 1))
        if y < self.sizey * TILES_PER_INCH - 1:
            neighbors.append((self.board[y + 1][x], 1))
        
        # check diagonals
        if x > 0 and y > 0:
            neighbors.append((self.board[y - 1][x - 1], math.sqrt(2)))
        if x < self.sizex * TILES_PER_INCH - 1 and y > 0:
            neighbors.append((self.board[y - 1][x + 1], math.sqrt(2)))
        if x > 0 and y < self.sizey * TILES_PER_INCH - 1:
            neighbors.append((self.board[y + 1][x - 1], math.sqrt(2)))
        if x < self.sizex * TILES_PER_INCH - 1 and y < self.sizey * TILES_PER_INCH - 1:
            neighbors.append((self.board[y + 1][x + 1], math.sqrt(2)))

        return neighbors


class p_Board(Board):
    def __init__(self, sizex: float, sizey: float, tile_queue: Queue, model_queue: Queue, pathfinder_queue: Queue, tile_return_queue: Queue, pathfinder_return_queue: Queue):
        super().__init__(sizex, sizey)

        self.tile_queue = tile_queue
        self.model_queue = model_queue
        self.pathfinder_queue = pathfinder_queue
        self.tile_return_queue = tile_return_queue
        self.pathfinder_return_queue = pathfinder_return_queue

        self.tiles_need_update = False

        self.board : list[list[p_Tile_wrapper]] = [[p_Tile_wrapper(p_Tile(i, j, True)) for i in range(sizex * TILES_PER_INCH)] for j in range(sizey * TILES_PER_INCH)]
        self.pathfinders : list[Pathfinder] = []
        self.models : list[Model] = []

        self.current_pathfinder = 0
        print("Process board started")
        self.update()  

    def update(self):

        print("Starting update")
        while(True):
            # update the tiles from the tile queue
            while not self.tile_queue.empty():
                tile = self.tile_queue.get()
                self.board[tile.y][tile.x].tile = tile
            
            # update the models from the model queue
            while not self.model_queue.empty():
                model = self.model_queue.get()
                # if the model is already in the list update the model otherwise add the model to the list
                model_found = False
                for i, m in enumerate(self.models):
                    if m.id == model[0]:
                        self.models[i].radius = model[1]
                        self.models[i].x = model[2]
                        self.models[i].y = model[3]
                        self.models[i].current_x = model[4]
                        self.models[i].current_y = model[5]
                        self.models[i].placed_x = model[6]
                        self.models[i].placed_y = model[7]
                        self.models[i].speed = model[8]
                        self.models[i].inspected = model[9]
                        self.models[i].show_vision = model[10]
                        self.models[i].vision_range = model[11]


                        model_found = True
                if not model_found:
                    model_obj = Model(self, model[0], model[1], model[2], model[3], model[8], (255, 0, 0, 255))
                    model_obj.current_x = model[4]
                    model_obj.current_y = model[5]
                    model_obj.placed_x = model[6]
                    model_obj.placed_y = model[7]
                    model_obj.inspected = model[9]
                    model_obj.show_vision = model[10]
                    model_obj.vision_range = model[11]
                    self.models.append(model_obj)
                    # add the model to each tile 


            # update the pathfinders from the pathfinder queue, pathfinders are sent in a tuple with the pathfinder type, model id, and a bool to determine if the pathfinder should be removed
            while not self.pathfinder_queue.empty():
                pathfinder = self.pathfinder_queue.get()
                if pathfinder[2]:
                    for i, p in enumerate(self.pathfinders):
                        if p.model.id == pathfinder[1]:
                            self.pathfinders.pop(i)
                else:
                    # if the pathfinder is already in the list update the pathfinder otherwise add the pathfinder to the list
                    pathfinder_found = False
                    for i, p in enumerate(self.pathfinders):
                        if p.model.id == pathfinder[1]:
                            self.pathfinders[i].requires_setup = True
                            pathfinder_found = True 
                    if not pathfinder_found:
                        if pathfinder[0] == "moving":
                            movement_pathfinder = moving_Pathfinder(self, self.models[pathfinder[1]])
                            self.pathfinders.append(movement_pathfinder)
                            movement_pathfinder.setup_pathfinding()
                        elif pathfinder[0] == "vision":
                            vision_pathfinder = vision_Pathfinder(self, self.models[pathfinder[1]])
                            self.pathfinders.append(vision_pathfinder)
                            vision_pathfinder.setup_pathfinding()
            
            # update the pathfinding for all models
            # update the pathfinding for all models
            if len(self.pathfinders) > 0:
                time_per_pathfinder = 0.04 / len(self.pathfinders)
            else:
                time_per_pathfinder = 0.04
            for i, pathfinder in enumerate(self.pathfinders[self.current_pathfinder:]):
                if pathfinder.update_pathfinding(time_per_pathfinder):
                    self.current_pathfinder = i  
                
                if pathfinder.pathfinding_done:
                    self.pathfinder_return_queue.put((pathfinder.model.id, pathfinder.pathfinding_type, pathfinder.tiles))
            
            self.current_pathfinder = 0

            
            # send the updated tiles to the tile return queue
            for row in self.board:
                for tile in row:
                    if tile.tile.updated:
                        self.tile_return_queue.put(tile.tile)
                        tile.tile.updated = False



class Gui_Board(Board):
    # board class size in inches, 10 tiles for every 1 inch
    def __init__(self, sizex: float, sizey: float):
        super().__init__(sizex, sizey)

        self.board : list[list[Tile]] = [[Tile(self ,i, j, True) for i in range(sizex * TILES_PER_INCH)] for j in range(sizey * TILES_PER_INCH)]
        
        
        self.models : list[Model] = []
        self.unwalkable_tiles = []
        self.model_id_counter = 0

        self.pathfinding_done = True
        self.pathfinding_setup = False

        self.view_port_origin = imgui.ImVec2(0, 0)
        self.view_port_size = imgui.ImVec2(sizex * TILES_PER_INCH, sizey * TILES_PER_INCH)
        # passd the right amount of pixels per tile to fit the board in the window
        self.window_size = imgui.ImVec2(800, 600)
        self.pixels_per_tile = min(self.window_size.x / self.view_port_size.x, self.window_size.y / self.view_port_size.y)
        self.ImageRgb = np.zeros((int(sizey * TILES_PER_INCH * self.pixels_per_tile), int(sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)
        self._ImageRgb = np.zeros((int(sizey * TILES_PER_INCH * self.pixels_per_tile), int(sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)    

        self.start_time = 0
        self.tile_time = 0
        self.model_time = 0
        self.render_time = 0
        self.pathfinding_time = 0

        self.glfw = None

        self.pathfinders : list[Pathfinder] = []

        self.tiles_need_update = False

        self._tiles_needs_update = False

        self.last_tile_update = time.time()

        self.tile_update_interval = 1.0

        self.tile_update_counter = (0, 0)

        self.pathfinder_update_counter = 0

        self.current_pathfinder = 0

        self.refresh_display = False

        # queue for sending tiles to the update process
        self.tile_queue = Queue()

        # queue for sending models to the update process
        self.model_queue = Queue()

        # queue for sending pathfinders to the update process
        self.pathfinder_queue = Queue()

        # queue for receiveing tiles from the update process
        self.tile_return_queue = Queue()

        # queue for receiving pathfinders from the update process
        self.pathfinder_return_queue = Queue()


        #self.tile_update_thread_active = True

        #self.tile_update_thread = threading.Thread(target=self.update_tiles_thread)
        #self.tile_update_thread.start()

        self.process_board = Process(target=p_Board, args=(self.sizex, self.sizey, self.tile_queue, self.model_queue, self.pathfinder_queue, self.tile_return_queue, self.pathfinder_return_queue))

        #self.process_board.start()



    # a function to create terrain on the board by setting the walkable status of the tiles given a 2d array of bools and a start position
    def add_terrain(self, terrain: list[list[bool]], startx: int, starty: int):
        # check that the terrain plis the start position fits on the board
        if len(terrain) + starty > self.sizey * TILES_PER_INCH or len(terrain[0]) + startx > self.sizex * TILES_PER_INCH:
            return
        
        # convert startx and starty to tiles
        startx *= TILES_PER_INCH
        starty *= TILES_PER_INCH

        # make startx and starty ints
        startx = int(startx)
        starty = int(starty)
        
        for y in range(len(terrain)):
            for x in range(len(terrain[y])):
                # if the tile is already unwalkable leave it alone otherwise set the tile to the value in the terrain array
                if not self.board[y + starty][x + startx].tile.is_walkable:
                    continue
                self.board[y + starty][x + startx].tile.is_walkable = terrain[y][x]

                # if the tile is unwalkable add it to the list of unwalkable tiles
                if not terrain[y][x]:
                    self.unwalkable_tiles.append(self.board[y + starty][x + startx])

    # static method to create a corner terrain shape
    @staticmethod
    def make_corner_terrain(width, height, corner):
            terrain = [[True for i in range(width)] for j in range(height)]
            if corner == "top left":
                for i in range(width):
                    for j in range(height):
                        # if i or j are the max value set the tile to unwalkable
                        if i == width - 1 or j == height - 1:
                            terrain[j][i] = False
            elif corner == "top right":
                for i in range(width):
                    for j in range(height):
                        # if i is 0 and j is max set the tile to unwalkable
                        if i == 0 or j == height - 1:
                            terrain[j][i] = False
                        
            elif corner == "bottom left":
                for i in range(width):
                    for j in range(height):
                        
                        # if i is max and j is 0 set the tile to unwalkable
                        if i == width - 1 or j == 0:
                            terrain[j][i] = False
            elif corner == "bottom right":
                for i in range(width):
                    for j in range(height):
                        # if i and j are 0 set the tile to unwalkable
                        if i == 0 or j == 0:
                            terrain[j][i] = False
            return terrain



    
    def add_model(self, radius: float, x: int, y: int):
        model = Model(self, self.model_id_counter, radius, x, y)

        self.models.append(model)
        self.model_id_counter += 1
        self.clear_reachable_tiles()

            
    def shutdown(self):
        try:
            self.process_board.terminate()
            self.process_board.join()
        except:
            pass
    
    def update(self):

        # draw the board as a grid offset from the cursor position
        # find the right amount of pixels per tile to fit the board in the window

        if self.glfw == None:
            self.glfw = glfw_utils.glfw_window_hello_imgui()
        
        
        # board + fps
        imgui.text("FPS: {:.2f}".format(1/imgui.get_io().delta_time))
        # total time
        imgui.text("Total time: {:.4f}".format(time.time() - self.start_time))
        # pathfinding update time
        imgui.text("Pathfinding update time: {:.4f}".format(self.pathfinding_time - self.start_time))
        # tile update time 
        imgui.text("Tile update time: {:.4f}".format(self.tile_time - self.pathfinding_time))
        # render time
        imgui.text("Render time: {:.4f}".format(self.render_time - self.tile_time))
        # model update time
        imgui.text("Model update time: {:.4f}".format(self.model_time - self.render_time))
        # extra time
        imgui.text("Extra time: {:.4f}".format(time.time() - self.model_time))

        self.start_time = time.time()

        # check if the window size has changed and update the pixels per tile
        
        new_window_size = imgui.get_content_region_avail() - imgui.ImVec2(0, 50)

        if new_window_size.x != self.window_size.x or new_window_size.y != self.window_size.y:
            self.window_size = new_window_size
            self.pixels_per_tile = min(self.window_size.x / self.view_port_size.x, self.window_size.y / self.view_port_size.y)
            self.ImageRgb = np.zeros((int(self.sizey * TILES_PER_INCH * self.pixels_per_tile), int(self.sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)

            if not self.tiles_need_update:
                self.tiles_need_update = True
            #self.tile_update_counter = (0, 0)

        cursor_pos = imgui.get_cursor_pos()

              
        # update the pathfinding for all models
        if len(self.pathfinders) > 0:
            time_per_pathfinder = 0.04 / len(self.pathfinders)
        else:
            time_per_pathfinder = 0.04

        pathfinding_done = True
        for i, pathfinder in enumerate(self.pathfinders):
            pathfinder.update_pathfinding(time_per_pathfinder)

        '''
        # send the updated tiles to the tile queue
        for row in self.board:
            for tile in row:
                if tile.tile.updated:
                    self.tile_queue.put(tile.tile)
                    tile.tile.updated = False
        
        # get the updated tiles from the tile return queue
        while not self.tile_return_queue.empty():
            tile = self.tile_return_queue.get()
            self.board[tile.y][tile.x].tile = tile

        # send the models to the model queue
        for model in self.models:
            if model.updated:
                # send the model id, radius, x, y, current_x, current_y, placed_x, placed_y and speed to the model queue
                cmd = (model.id, model.radius, model.x, model.y, model.current_x, model.current_y, model.placed_x, model.placed_y, model.speed, model.inspected, model.show_vision, model.vision_range)
                self.model_queue.put(cmd)
                model.updated = False

        # get the updated pathfinders from the pathfinder return queue
        while not self.pathfinder_return_queue.empty():
            pathfinder = self.pathfinder_return_queue.get()
            for i, p in enumerate(self.pathfinders):
                if p.model.id == pathfinder[0] and p.pathfinder_type == pathfinder[1]:
                    self.pathfinders[i].tiles = pathfinder[2]
                    self.pathfinders[i].requires_setup = False
                    break
        '''

        self.pathfinding_time = time.time()




        #if self.tiles_need_update:
            # update the tiles if the time since the last update is greater than the update interval
        #if time.time() - self.last_tile_update > self.tile_update_interval:
        if self.update_tiles():
            self.tile_update_counter = (0, 0)
            self.tiles_need_update = False
            print("tiles updated in ", time.time() - self.last_tile_update)
            self.last_tile_update = time.time()
            self._ImageRgb = self.ImageRgb.copy()
            self.refresh_display = True

        self.tile_time = time.time()

        
        immvision.image(
        "##Original", self._ImageRgb, immvision.ImageParams(show_zoom_buttons=False, show_pixel_info=False, show_options_button=False, can_resize=False, refresh_image=self.refresh_display))

        self.refresh_display = False

        self.render_time = time.time()

        '''
        for model in self.models:
            state, x, y = model.update(cursor_pos)

            if state == ModelState.PICKED_UP:
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                # send the pathfinder details to the pathfinder queue
                self.pathfinder_queue.put(("moving", model.id, False))
                self.pathfinder_queue.put(("vision", model.id, False))
                
                break
            elif state == ModelState.PLACED:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                self.pathfinder_queue.put(("moving", model.id, True))
                self.pathfinder_queue.put(("vision", model.id, True))
                # reapply the pathfinding for all models
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.INSPECTED:
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                self.pathfinder_queue.put(("moving", model.id, False))
                self.pathfinder_queue.put(("vision", model.id, False))
                break
            elif state == ModelState.UNINSPECTED:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                # clear the pathfinding for the model
                self.pathfinder_queue.put(("moving", model.id, True))
                self.pathfinder_queue.put(("vision", model.id, True))
                # restart the pathfinding for all models
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.REFRESH:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                self.pathfinder_queue.put(("moving", model.id, False))
                self.pathfinder_queue.put(("vision", model.id, False))
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.LETGO:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                self.pathfinder_queue.put(("moving", model.id, False))
                self.pathfinder_queue.put(("vision", model.id, False))
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            '''
        for model in self.models:
            state, x, y = model.update(cursor_pos)

            if state == ModelState.PICKED_UP:
                pathfinder_found = False
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)

                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        pathfinder.requires_setup = True
                        pathfinder_found = True
                if not pathfinder_found:
                    movement_pathfinder = moving_Pathfinder(self, model)
                    self.pathfinders.append(movement_pathfinder)
                    movement_pathfinder.setup_pathfinding()
                    vision_pathfinder = vision_Pathfinder(self, model)
                    self.pathfinders.append(vision_pathfinder)
                    vision_pathfinder.setup_pathfinding()
                
                break
            elif state == ModelState.PLACED:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                # clear the pathfinding for the model
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        self.pathfinders.remove(pathfinder)
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        self.pathfinders.remove(pathfinder)
                # reapply the pathfinding for all models
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.INSPECTED:
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                pathfinder_found = False
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        pathfinder.requires_setup = True
                        pathfinder_found = True
                if not pathfinder_found:
                    movement_pathfinder = moving_Pathfinder(self, model)
                    self.pathfinders.append(movement_pathfinder)
                    movement_pathfinder.setup_pathfinding()
                    vision_pathfinder = vision_Pathfinder(self, model)
                    self.pathfinders.append(vision_pathfinder)
                    vision_pathfinder.setup_pathfinding()
                break
            elif state == ModelState.UNINSPECTED:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                # clear the pathfinding for the model
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        self.pathfinders.remove(pathfinder)
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        self.pathfinders.remove(pathfinder)
                # restart the pathfinding for all models
                for pathfinder in self.pathfinders:
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.REFRESH:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        pathfinder.requires_setup = True
                    pathfinder.tiles_need_applied = True
                break
            elif state == ModelState.LETGO:
                self.clear_reachable_tiles()
                #if not self.tiles_need_update:
                self.tiles_need_update = True
                self.tile_update_counter = (0, 0)
                for pathfinder in self.pathfinders:
                    if pathfinder.model.id == model.id:
                        #pathfinder.remove_tiles()
                        pathfinder.requires_setup = True
                    pathfinder.tiles_need_applied = True
                break

                

        
        self.model_time = time.time()
        
        # set the cursor position to the row after the end of the board
        imgui.set_cursor_pos(imgui.ImVec2(cursor_pos.x + self.sizex * TILES_PER_INCH * self.pixels_per_tile, cursor_pos.y + self.sizey * TILES_PER_INCH * self.pixels_per_tile))

   
    

    def update_tiles(self):
        start_time = time.time()

        for i, row in enumerate(self.board[self.tile_update_counter[0]:]):
            if i == 0:
                for tile in row[self.tile_update_counter[1]:]:
                    tile.update(self.pixels_per_tile)
                    if time.time() > start_time + 0.005:
                        self.tile_update_counter = (i + self.tile_update_counter[0], j + self.tile_update_counter[1])
                        return False
            else:
                for j, tile in enumerate(row):
                    tile.update(self.pixels_per_tile)
                    # if the current time is greater than the start time + 0.01 seconds return
                    if time.time() > start_time + 0.01:
                        self.tile_update_counter = (i + self.tile_update_counter[0], j)
                        return False
        self.tile_update_counter = (0, 0)
        return True

    def clear_reachable_tiles(self):
        print("clearing reachable tiles")
        for row in self.board:
            for tile in row:
                tile.needs_reset = True


        self.ImageRgb = np.zeros((int(self.sizey * TILES_PER_INCH * self.pixels_per_tile), int(self.sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)





# an enum representing the states of a model 
class ModelState:
    IDLE = 0
    MOVING = 1
    PICKED_UP = 2
    PLACED = 3
    INSPECTED = 4
    UNINSPECTED = 5
    REFRESH = 6
    LETGO = 7



class Model: 
    def __init__(self, board: Board, id: int, radius: float, x: int, y: int, speed: float = 12 ,color: tuple[int, int, int, int] = (255, 0, 0, 255)):
        # x and y are the center position of the model passed in inches, and stored in tiles, radius is the radius of the model passed in inches and stored in tiles
        self.id = id
        self.radius = radius * TILES_PER_INCH
        self.x = x * TILES_PER_INCH
        self.y = y * TILES_PER_INCH
        # speed passed in inches, stored in tiles
        self.speed = speed * TILES_PER_INCH
        self.board = board
        self.moving = False
        self.placed = True
        self.inspected = False
        self.last_mouse_pos = imgui.ImVec2(0, 0)
        self.color = color
        self.state = ModelState.IDLE
        self.check_valid_position(check_reachable=False)
        self.current_x = self.x
        self.current_y = self.y
        self.placed_x = self.x
        self.placed_y = self.y
        self.previous_delta = imgui.ImVec2(0, 0)
        
        self.vision_range = 24 * TILES_PER_INCH

        self.show_vision = False

        self.updated = True

        
    
    def update(self, cursor_pos):

        # if the model is right clicked, open the menu
        if imgui.is_mouse_clicked(1) and imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile, 
                                                                               cursor_pos.y + self.y * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile), 
                                                                      imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile, 
                                                                               cursor_pos.y + self.y * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile)):
            # open the menu as a popup window
            imgui.open_popup("Model Menu##" + str(self.id))

            
        
        menu_open = imgui.begin_popup("Model Menu##" + str(self.id))

        if self.inspected:
            # draw a circle around the model to represent the speed
            imgui.get_window_draw_list().add_circle(imgui.ImVec2(cursor_pos.x + self.placed_x * self.board.pixels_per_tile, cursor_pos.y + self.placed_y * self.board.pixels_per_tile), (self.speed + self.radius) * self.board.pixels_per_tile, imgui.get_color_u32((0, 0, 255, 128)), num_segments=64)

        if self.show_vision:
            imgui.get_window_draw_list().add_circle(imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile, cursor_pos.y + self.y * self.board.pixels_per_tile), self.vision_range * self.board.pixels_per_tile, imgui.get_color_u32((255, 0, 0, 128)), num_segments=64)

        if menu_open:
            # controls to edit the radius, speed, and color of the model. plus a button to place the model
            #_, self.radius = imgui.slider_float("Radius##slider", self.radius / TILES_PER_INCH, 0.1, 5)
            # add a float field to manually set the radius
            changed_radius , self.radius = imgui.input_float("Radius##input", self.radius / TILES_PER_INCH, 0.1, 1, "%.1f")
            #_, self.speed = imgui.slider_float("Speed##slider", self.speed / TILES_PER_INCH, 0.1, 5)
            # add a float field to manually set the speed
            changed_speed, self.speed = imgui.input_float("Speed##input", self.speed / TILES_PER_INCH, 0.5, 1, "%.1f")

            changed_vision_range, self.vision_range = imgui.input_float("Vision Range##input", self.vision_range / TILES_PER_INCH, 0.5, 1, "%.1f")

            self.vision_range *= TILES_PER_INCH

            changed_vis, self.show_vision = imgui.checkbox("Show Vision", self.show_vision)
            
            #_, self.color = imgui.color_edit4("Color", *self.color)

            # convert the radius and speed back to tiles
            self.radius *= TILES_PER_INCH
            self.speed *= TILES_PER_INCH

            # if the radius or speed has changed return 
            if changed_radius or changed_speed or changed_vision_range or changed_vis: 
                imgui.end_popup()

                # check the position of the model is valid
                self.check_valid_position()
                self.current_x = self.x
                self.current_y = self.y
                self.updated = True

                # set the state to uninspected to update the pathfinding     
                self.state = ModelState.REFRESH
                return (self.state, self.current_x, self.current_y)


            if imgui.button("Place"):
                self.state = ModelState.PLACED
                self.placed = True
                self.placed_x = self.x
                self.placed_y = self.y
                self.inspected = False
                self.updated = True
                imgui.end_popup()
                return (self.state, self.current_x, self.current_y)
            imgui.end_popup()

        

        if self.moving:
            imgui.set_mouse_cursor(imgui.MouseCursor_.none)
            # get the center of the screen in pixels offset by the cursor position
            screen_center = imgui.ImVec2(cursor_pos.x + 2 * self.board.sizex * self.board.pixels_per_tile, cursor_pos.y + 2 * self.board.sizey * self.board.pixels_per_tile)
            
            
            # get the current mouse position
            current_mouse_pos = glfw_utils.glfw.get_cursor_pos(self.board.glfw)
            current_mouse_pos = imgui.ImVec2(current_mouse_pos[0], current_mouse_pos[1])

            # calculate the delta of the mouse position from the center of the screen
            delta = current_mouse_pos - screen_center

            # if the current delta is identical to the previous delta set the delta to 0
            if delta == self.previous_delta:
                delta = imgui.ImVec2(0, 0)
            else:
                self.previous_delta = delta

            # move the mouse back by a small amount to keep the cursor in the center of the screen
            glfw_utils.glfw.set_cursor_pos(self.board.glfw, screen_center.x, screen_center.y)
            
            


            # if delta is greater than radius of the model clamp the delta to the radius of the model
            scaler = 5 
            if delta.x ** 2 + delta.y ** 2 > (self.radius * scaler) ** 2:
                angle = math.atan2(delta.y, delta.x)
                delta = imgui.ImVec2(math.cos(angle) * self.radius * scaler, math.sin(angle) * self.radius * scaler)
 
            self.x += delta.x / self.board.pixels_per_tile
            self.y += delta.y / self.board.pixels_per_tile

            check, dir = self.check_collision()
            if check:
                # if both x and y are colliding move the model to the closest point where it is not colliding
                if "X" in dir and "Y" in dir:
                    self.check_valid_position()
                    pass 
                elif "X" in dir:
                    self.x -= delta.x / self.board.pixels_per_tile
                elif "Y" in dir:
                    self.y -= delta.y / self.board.pixels_per_tile

                   

            self.state = ModelState.MOVING

            # if mouse is released check if the model is in bounds
            if imgui.is_mouse_released(0):
                self.moving = False
                self.check_valid_position()
                self.current_x = self.x
                self.current_y = self.y
                
                # if the hold time is greater than 0.1 seconds the model is picked up
                if time.time() - self.hold_time_start < 0.2:
                    if not self.inspected:
                        self.state = ModelState.UNINSPECTED
                else:
                    self.inspected = True
                    self.state = ModelState.LETGO
                    self.updated = True


                # update the mouse position to the current position in pixels
                glfw_utils.glfw.set_cursor_pos(self.board.glfw, cursor_pos.x + self.x * self.board.pixels_per_tile, cursor_pos.y + self.y * self.board.pixels_per_tile)
        else:
            if imgui.is_mouse_clicked(0) and imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile, 
                                                                                    cursor_pos.y + self.y * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile), 
                                                                        imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile, 
                                                                                    cursor_pos.y + self.y * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile)):
                
                # set the cursor position to the center of the screen
                self.hold_time_start = time.time()
                imgui.set_mouse_cursor(imgui.MouseCursor_.none)
                screen_center = imgui.ImVec2(cursor_pos.x + 2 * self.board.sizex * self.board.pixels_per_tile, cursor_pos.y + 2 * self.board.sizey * self.board.pixels_per_tile)
                glfw_utils.glfw.set_cursor_pos(self.board.glfw, screen_center.x, screen_center.y)
                glfw_utils.glfw.poll_events()
                self.moving = True
                if self.placed:
                    self.state = ModelState.PICKED_UP
                    print(f"picking up model at {self.x}, {self.y}")
                    self.placed = False
                    self.inspected = True
                else:
                    if self.inspected:
                        self.inspected = False
                        self.state = ModelState.MOVING
                    else:
                        self.state = ModelState.INSPECTED
                        self.inspected = True
            else:
                self.state = ModelState.IDLE



        # convert the x and y position to pixels
        _x = cursor_pos.x   + self.x  * self.board.pixels_per_tile
        _y = cursor_pos.y  + self.y  * self.board.pixels_per_tile
        
        imgui.get_window_draw_list().add_circle_filled(imgui.ImVec2(_x, _y), self.radius * self.board.pixels_per_tile, imgui.get_color_u32((255, 0, 0, 255)), num_segments=64)

        # draw text with the distance of the model from the start tile
        try:
            if self.board.board[int(self.y)][int(self.x)].tile.distance[self.id] != None:
                imgui.get_window_draw_list().add_text(imgui.ImVec2(_x - self.radius * self.board.pixels_per_tile / 2, _y - self.radius * self.board.pixels_per_tile / 2), imgui.get_color_u32((0, 0, 0, 255)), f"{round(self.board.board[int(self.y)][int(self.x)].tile.distance[self.id] / TILES_PER_INCH, 1)}")
        except:
            pass

        return (self.state, self.current_x, self.current_y)
    
    def check_collision(self):
        # check of the model is colliding with the edge of the board
        out = False
        dir = []

        if self.x - self.radius < 0:
            self.x = self.radius
        if self.x + self.radius >= self.board.sizex * TILES_PER_INCH:
            self.x = self.board.sizex * TILES_PER_INCH - self.radius

        if self.y - self.radius < 0:
            self.y = self.radius
        if self.y + self.radius >= self.board.sizey * TILES_PER_INCH:
            self.y = self.board.sizey * TILES_PER_INCH - self.radius
        
        
        # check if the model is outside the speed of the model
        distance = math.sqrt((self.x - self.current_x) ** 2 + (self.y - self.current_y) ** 2)
        if distance > self.speed + self.radius:
            out = True
            dir.append("X")
            dir.append("Y")

        # check if the model is colliding with an unwalkable tile
        for row in self.board.board:
            for tile in row:
                tile = tile.tile
                if not tile.is_walkable or tile.edge:
                    distance = math.sqrt((self.x - tile.x) ** 2 + (self.y - tile.y) ** 2)
                    if distance < self.radius:
                        # find out if the colliding tile is on the x or y axis
                        if self.x - tile.x < self.radius:
                            dir.append("X")
                        if self.y - tile.y < self.radius:
                            dir.append("Y")
                        out = True

        # check if the model is colliding with another model
        for model in self.board.models:
            if model != self:
                distance = math.sqrt((self.x - model.x) ** 2 + (self.y - model.y) ** 2)
                if distance < self.radius + model.radius:
                    # if the model is colliding with another model move the model to the closest point where it is not colliding
                    angle = math.atan2(self.y - model.y, self.x - model.x)
                    self.x = model.x + math.cos(angle) * (self.radius + model.radius)
                    self.y = model.y + math.sin(angle) * (self.radius + model.radius)

        
        
        return out, dir



    def check_valid_position(self, check_reachable: bool = True):

        # check if the model is colliding with the edge of the board
        if self.x - self.radius < 0:
            self.x = self.radius
        if self.x + self.radius > self.board.sizex * TILES_PER_INCH:
            self.x = self.board.sizex * TILES_PER_INCH - self.radius
        if self.y - self.radius < 0:
            self.y = self.radius
        if self.y + self.radius > self.board.sizey * TILES_PER_INCH:
            self.y = self.board.sizey * TILES_PER_INCH - self.radius

        # check if the model is colliding with another model
        for model in self.board.models:
            if model != self:
                distance = math.sqrt((self.x - model.x) ** 2 + (self.y - model.y) ** 2)
                if distance < self.radius + model.radius:
                    # if the model is colliding with another model move the model to the closest point where it is not colliding
                    angle = math.atan2(self.y - model.y, self.x - model.x)
                    self.x = model.x + math.cos(angle) * (self.radius + model.radius)
                    self.y = model.y + math.sin(angle) * (self.radius + model.radius)
        

        # check if the model is colliding with an unwalkable tile
        for row in self.board.board:
            for tile in row:
                tile = tile.tile
                if not tile.is_walkable:
                    distance = math.sqrt((self.x - tile.x) ** 2 + (self.y - tile.y) ** 2)
                    if distance < self.radius:
                        # if the model is colliding with an unwalkable tile move the model to the closest point where it is not colliding
                        angle = math.atan2(self.y - tile.y, self.x - tile.x)
                        self.x = tile.x + math.cos(angle) * self.radius
                        self.y = tile.y + math.sin(angle) * self.radius
        
        self.x = round(self.x)
        self.y = round(self.y)

        if check_reachable:
            self.check_reachable()
        
    
    def check_reachable(self):

        # check if the current position of the model is reachable by checking the distance of the tile to the models speed - the radius of the model
        try:
            if self.board.board[self.y][self.x].tile.distance[self.id] == None:
                self.move_to_closest_valid() 
        except:
            self.move_to_closest_valid()
        
        try:
            if self.board.board[self.y][self.x].tile.distance[self.id] >= self.speed:
                # if not find the closest reachable tile
                self.move_to_closest_valid()
        except:
            self.move_to_closest_valid()
    
    def move_to_closest_valid(self):
        closest_distance = None
        closest_tile = None
        for row in self.board.board:
            for tile in row:
                try:
                    if tile.distance[self.id] != None and tile.distance[self.id] < self.speed:
                        distance = math.sqrt((self.x - tile.x) ** 2 + (self.y - tile.y) ** 2)
                        if closest_distance == None or distance < closest_distance:
                            closest_distance = distance
                            closest_tile = tile
                except:
                    pass
        if closest_tile != None:
            self.x = closest_tile.x
            self.y = closest_tile.y
        else:
            self.x = self.current_x
            self.y = self.current_y
        
        self.check_valid_position(check_reachable=False)


class Pathfinder:
    def __init__(self, board: Board, model: Model):
        self.board = board
        self.model = model
        self.requires_setup = True
        self.tiles_need_applied = False
        self.tiles = []

    def setup_pathfinding(self):
        if self.board.tiles_need_update:
            return True
        self.requires_setup = False
        return False

    def update_pathfinding(self, update_time):
        if self.requires_setup:
            if self.setup_pathfinding():
                return True
        if self.board.tiles_need_update:
            return True
        if self.tiles_need_applied:
            if self.apply_tiles():
                return True
        return False
    
    def apply_tiles(self):
        pass

    def remove_tiles(self):
        pass


class vision_Pathfinder(Pathfinder):
    def __init__(self, board: Board, model: Model):
        super().__init__(board, model)
        self.pathfinding_type = "vision"

    def setup_pathfinding(self):
        if super().setup_pathfinding():
            return True
        
        self.tiles = []
        # a flood fill pathfinding, that gets the neightbors of the current tile and checks if they are reachable and stores the distance from the start tile
        # then the neighbors of the neighbors are checked and so on until the distance is greater than the speed of the model
        self.x = self.model.current_x
        self.y = self.model.current_y

        # ensure x and y are ints and in bounds
        self.x = min(max(self.x, 0), self.board.sizex * TILES_PER_INCH - 1)
        self.y = min(max(self.y, 0), self.board.sizey * TILES_PER_INCH - 1)

        print(f"setting up pathfinding at {self.x}, {self.y}")
        # speed adjusted to radius of the model
        self.speed = self.model.vision_range
        self.open_list = []

        # set the rest of the tiles to unreachable
        for row in self.board.board:
            for tile in row:
                tile.tile.set_visible_state(self.model.id, False)
                tile.tile.visibility_distance[self.model.id] = None


        self.board.board[self.y][self.x].tile.set_visible_state(self.model.id, True)
        self.board.board[self.y][self.x].tile.visibility_distance[self.model.id] = 0

        self.tiles.append((self.x, self.y, 0))
        

        #self.open_list.append((x, y, 0))
        # add the neighbors of the start tile to the open list
        for neighbor, distance in self.board.get_neighbors(self.x, self.y):
            if neighbor.tile.is_walkable:
                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance))

        self.pathfinding_setup = True
        self.pathfinding_done = False
    
    def is_visible(self, x, y):
        # check if the tile is visible by checking if there are any unwalkable tiles between the model and the tile
        '''
        offset_x = 0
        offset_y = 0

        model_x = self.model.current_x + offset_x
        model_y = self.model.current_y + offset_y
        
        angle = math.atan2(y - model_y, x - model_x) 
        distance = math.sqrt((x - model_x) ** 2 + (y - model_y) ** 2)
        good_path = True
        for k in range(1, int(distance)):
            x2 = model_x + (math.cos(angle) * k)
            y2 = model_y + (math.sin(angle) * k)
            # clamp the x and y values to the size of the board
            x2 = min(max(x2, 0), self.board.sizex * TILES_PER_INCH - 1)
            y2 = min(max(y2, 0), self.board.sizey * TILES_PER_INCH - 1)

            if not self.board.board[int(y2)][int(x2)].is_walkable:
                good_path = False
                break
        if good_path:
            return True
        '''
        
        
        offset_x = 0
        offset_y = self.model.radius

        skip = False

        # check if the x, y poisiton is in the direction of the offset from the model
        if y < self.model.current_y - offset_y:
            skip = True
        
        model_x = self.model.current_x + offset_x
        model_y = self.model.current_y + offset_y

        if not skip:
            angle = math.atan2(y - model_y, x - model_x)
            distance = math.sqrt((x - model_x) ** 2 + (y - model_y) ** 2)
            good_path = True
            for k in range(1, int(distance)):
                x2 = model_x + (math.cos(angle) * k)
                y2 = model_y + (math.sin(angle) * k)
                # clamp the x and y values to the size of the board
                x2 = min(max(x2, 0), self.board.sizex * TILES_PER_INCH - 1)
                y2 = min(max(y2, 0), self.board.sizey * TILES_PER_INCH - 1)

                if not self.board.board[int(y2)][int(x2)].tile.is_walkable:
                    good_path = False
                    break
            if good_path:
                return True

        offset_x = self.model.radius
        offset_y = 0

        skip = False

        # check if the x, y poisiton is in the direction of the offset from the model
        if x < self.model.current_x - offset_x:
            skip = True

        
        model_x = self.model.current_x + offset_x
        model_y = self.model.current_y + offset_y

        
        if not skip:
            angle = math.atan2(y - model_y, x - model_x) 
            distance = math.sqrt((x - model_x) ** 2 + (y - model_y) ** 2)
            good_path = True
            for k in range(1, int(distance)):
                x2 = model_x + (math.cos(angle) * k)
                y2 = model_y + (math.sin(angle) * k)
                # clamp the x and y values to the size of the board
                x2 = min(max(x2, 0), self.board.sizex * TILES_PER_INCH - 1)
                y2 = min(max(y2, 0), self.board.sizey * TILES_PER_INCH - 1)

                if not self.board.board[int(y2)][int(x2)].tile.is_walkable:
                    good_path = False
                    break
            if good_path:
                return True
            
        
        offset_x = 0
        offset_y = -self.model.radius

        skip = False

         # check if the x, y poisiton is in the direction of the offset from the model
        if y > self.model.current_y - offset_y:
            skip = True


        model_x = self.model.current_x + offset_x
        model_y = self.model.current_y + offset_y

        if not skip:
        
            angle = math.atan2(y - model_y, x - model_x) 
            distance = math.sqrt((x - model_x) ** 2 + (y - model_y) ** 2)
            good_path = True
            for k in range(1, int(distance)):
                x2 = model_x + (math.cos(angle) * k)
                y2 = model_y + (math.sin(angle) * k)
                # clamp the x and y values to the size of the board
                x2 = min(max(x2, 0), self.board.sizex * TILES_PER_INCH - 1)
                y2 = min(max(y2, 0), self.board.sizey * TILES_PER_INCH - 1)

                if not self.board.board[int(y2)][int(x2)].tile.is_walkable:
                    good_path = False
                    break
            if good_path:
                return True
        

        offset_x = -self.model.radius
        offset_y = 0

        skip = False

        # check if the x, y poisiton is in the direction of the offset from the model
        if x > self.model.current_x - offset_x:
            skip = True

        model_x = self.model.current_x + offset_x
        model_y = self.model.current_y + offset_y

        if not skip:
        
            angle = math.atan2(y - model_y, x - model_x) 
            distance = math.sqrt((x - model_x) ** 2 + (y - model_y) ** 2)
            good_path = True
            for k in range(1, int(distance)):
                x2 = model_x + (math.cos(angle) * k)
                y2 = model_y + (math.sin(angle) * k)
                # clamp the x and y values to the size of the board
                x2 = min(max(x2, 0), self.board.sizex * TILES_PER_INCH - 1)
                y2 = min(max(y2, 0), self.board.sizey * TILES_PER_INCH - 1)

                if not self.board.board[int(y2)][int(x2)].tile.is_walkable:
                    good_path = False
                    break
            if good_path:
                return True
            
        return False
    
    def update_pathfinding(self, update_time):
        if super().update_pathfinding(update_time):
            return False
        
        if not self.model.show_vision:
            return False
        
        start_time = time.time()
        
        if not self.pathfinding_done and self.pathfinding_setup:
            while len(self.open_list) > 0:
                x, y, distance = self.open_list.pop(0)
                if distance < self.speed:
                    if self.is_visible(x, y):
                        self.board.board[y][x].tile.set_visible_state(self.model.id, True)
                        self.tiles.append((x, y, distance))
                    for neighbor, neighbor_distance in self.board.get_neighbors(x, y):
                        if neighbor.tile.is_walkable:
                            # check if the neighbor is reachable and  if so update the distance and readd the neighbor to the open list
                            if neighbor.tile.visibility_distance[self.model.id] == None:
                                neighbor.tile.visibility_distance[self.model.id] = distance + neighbor_distance
                                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance + neighbor_distance))
                            elif distance + neighbor_distance < neighbor.tile.visibility_distance[self.model.id]:
                                neighbor.tile.visibility_distance[self.model.id] = distance + neighbor_distance
                                neighbor.tile.edge[self.model.id] = False
                                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance + neighbor_distance))
                if time.time() - start_time > update_time:
                    return True
            if len(self.open_list) == 0:
                self.pathfinding_done = True
        # return     
        return False
    
    def apply_tiles(self):
        if self.board.tiles_need_update:
            return True
        if self.requires_setup:
            self.tiles_need_applied = False
            return True

        for tile in self.tiles:
            self.board.board[tile[1]][tile[0]].tile.set_visible_state(self.model.id, True)
            self.board.board[tile[1]][tile[0]].tile.visibility_distance[self.model.id] = tile[2]
        self.tiles_need_applied = False
        return False
    
    def remove_tiles(self):
        if self.board.tiles_need_update:
            self.board.tile_update_counter = (0, 0)
            return True
        if self.requires_setup:
            self.tiles_need_applied = False
            return True

        for tile in self.tiles:
            self.board.board[tile[1]][tile[0]].tile.set_visible_state(self.model.id, False)
            self.board.board[tile[1]][tile[0]].tile.visibility_distance[self.model.id] = None
        self.tiles_need_applied = False
        return False


class moving_Pathfinder(Pathfinder):
    def __init__(self, board: Board, model: Model):
        self.pathfinding_type = "moving"
        super().__init__(board, model)
        
    def setup_pathfinding(self):
        if super().setup_pathfinding():
            return
        
        self.tiles = []
        # a flood fill pathfinding, that gets the neightbors of the current tile and checks if they are reachable and stores the distance from the start tile
        # then the neighbors of the neighbors are checked and so on until the distance is greater than the speed of the model
        self.x = self.model.placed_x
        self.y = self.model.placed_y
        print(f"setting up pathfinding at {self.x}, {self.y}")
        # speed adjusted to radius of the model
        self.speed = self.model.speed + self.model.radius
        self.open_list = []

        
        # set the rest of the tiles to unreachable
        for row in self.board.board:
            for tile in row:
                tile = tile.tile
                tile.set_reachable_state(self.model.id, False)
                tile.distance[self.model.id] = None
                tile.edge[self.model.id] = False
        

        self.board.board[self.y][self.x].tile.set_reachable_state(self.model.id, True)
        self.board.board[self.y][self.x].tile.distance[self.model.id] = 0

        self.tiles.append((self.x, self.y, 0))
        

        #self.open_list.append((x, y, 0))
        # add the neighbors of the start tile to the open list
        for neighbor, distance in self.board.get_neighbors(self.x, self.y):
            if neighbor.tile.is_walkable:
                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance))

        self.pathfinding_setup = True
        self.pathfinding_done = False
    
    def update_pathfinding(self, update_time):
        if super().update_pathfinding(update_time):
            return False
        changed_tiles = False

        start_time = time.time()

        if not self.pathfinding_done and self.pathfinding_setup:
            while len(self.open_list) > 0:
                x, y, distance = self.open_list.pop(0)
                if distance < self.speed:
                    self.board.board[y][x].tile.set_reachable_state(self.model.id, True)
                    self.tiles.append((x, y, distance))
                    changed_tiles = True
                    for neighbor, neighbor_distance in self.board.get_neighbors(x, y):
                        if neighbor.tile.is_walkable:
                            # check if the neighbor is reachable and  if so update the distance and readd the neighbor to the open list
                            if neighbor.tile.distance[self.model.id] == None:
                                neighbor.tile.distance[self.model.id] = distance + neighbor_distance
                                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance + neighbor_distance))
                            elif distance + neighbor_distance < neighbor.tile.distance[self.model.id]:
                                neighbor.tile.distance[self.model.id] = distance + neighbor_distance
                                neighbor.tile.edge[self.model.id] = False
                                self.open_list.append((neighbor.tile.x, neighbor.tile.y, distance + neighbor_distance))
                else:
                    self.board.board[y][x].tile.edge[self.model.id] = True
                if time.time() - start_time > update_time:
                    break
            if len(self.open_list) == 0:
                self.pathfinding_done = True    
        return changed_tiles
    
    def apply_tiles(self):
        if self.board.tiles_need_update:
            return True
        if self.requires_setup:
            self.tiles_need_applied = False
            return True

        for tile in self.tiles:
            self.board.board[tile[1]][tile[0]].tile.set_reachable_state(self.model.id, True)
            self.board.board[tile[1]][tile[0]].tile.distance[self.model.id] = tile[2]
        self.tiles_need_applied = False
        return False
    
    def remove_tiles(self):
        if self.board.tiles_need_update:
            return True
        if self.requires_setup:
            self.tiles_need_applied = False
            return True

        for tile in self.tiles:
            self.board.board[tile[1]][tile[0]].tile.set_reachable_state(self.model.id, False)
            self.board.board[tile[1]][tile[0]].tile.distance[self.model.id] = None
        self.tiles_need_applied = False
        return False



class p_Tile:
    def __init__(self, x: int, y: int, is_walkable: bool):
        self.x = x
        self.y = y
        self.is_walkable = is_walkable
        self.is_reachable = {int: bool}
        self.is_visible = {int: bool}
        self.is_any_reachable = False
        self.is_any_visible = False
        self.updated = True

        self.edge = {int: bool}
        self.distance = {int : float}
        self.visibility_distance = {int: float}

        self.already_clicked_without_break = False
        self.needs_reset = False

    def set_reachable_state(self, id: int, state: bool):
        self.is_reachable[id] = state
        if state:
            self.is_any_reachable = True
        elif not any(self.is_reachable.values()):
            self.is_any_reachable = False
        self.updated = True
    
    def set_visible_state(self, id: int, state: bool):
        self.is_visible[id] = state
        if state:
            self.is_any_visible = True
        elif not any(self.is_visible.values()):
            self.is_any_visible = False
        self.updated = True

    
class p_Tile_wrapper:
    def __init__(self, tile: p_Tile):
        self.tile = tile


class Tile:
    def __init__(self, board: Board ,x: int, y: int, is_walkable: bool, color: tuple[int, int, int, int] = (128, 128, 128, 255)):
        self.board = board
        self.tile = p_Tile(x, y, is_walkable)
        self.color = color

    def set_reachable_state(self, id: int, state: bool):
        self.tile.set_reachable_state(id, state)
    
    def set_visible_state(self, id: int, state: bool):
        self.tile.set_visible_state(id, state)

    

    
    def update(self, pixels_per_tile):
        # right click to change the walkable status of the tile
        if self.tile.needs_reset:
            self.tile.needs_reset = False
            self.tile.is_any_reachable = False
            self.tile.is_any_visible = False
            for key in self.tile.is_reachable:
                self.tile.is_reachable[key] = False
            for key in self.tile.is_visible:
                self.tile.is_visible[key] = False
            for key in self.tile.distance:
                self.tile.distance[key] = None
            for key in self.tile.visibility_distance:
                self.tile.visibility_distance[key] = None
            for key in self.tile.edge:
                self.tile.edge[key] = False
            self.updated = True
        
        '''
        if imgui.is_mouse_down(1) and imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * pixels_per_tile, cursor_pos.y + self.y * pixels_per_tile), imgui.ImVec2(cursor_pos.x + self.x * pixels_per_tile + pixels_per_tile, cursor_pos.y + self.y * pixels_per_tile + pixels_per_tile)):
            if not self.already_clicked_without_break:
                self.already_clicked_without_break = True
                self.is_walkable = not self.is_walkable
                # change the color of the tile to represent the walkable status
        else:
            self.already_clicked_without_break = False
        '''
        
        if self.tile.is_walkable:
            # if any value in the is_reachable dictionary is true set the color to blue
            self.color = (0, 0, 0, 0)
            if self.tile.is_any_reachable:
                self.color = (0, 0, 255, 255)
                if self.tile.is_any_visible:
                    self.color = (0, 255, self.color[2], 255)
            else:
                if self.tile.is_any_visible:
                    self.color = (0, 255, 0, 255)
                else:
                    self.color = (128, 128, 128, 255)
                    return
        else:
            self.color = (255, 0, 0, 255)

        _x = self.tile.x * pixels_per_tile
        _y = self.tile.y * pixels_per_tile

        # draw the tile in board.ImageRgb if the color is not the same as the current color
        
        self.board.ImageRgb[int(_y):int(_y + pixels_per_tile), int(_x):int(_x + pixels_per_tile)] = self.color[:3]

        #imgui.get_window_draw_list().add_rect_filled(imgui.ImVec2(_x, _y), imgui.ImVec2(_x + pixels_per_tile, _y + pixels_per_tile), imgui.get_color_u32(self.color))
        #imgui.get_window_draw_list().add_rect(imgui.ImVec2(_x, _y), imgui.ImVec2(_x + pixels_per_tile, _y + pixels_per_tile), imgui.get_color_u32((0, 0, 0, 255)))
        