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

TILES_PER_INCH = 5

immvision.use_rgb_color_order()

class Board:
    # board class size in inches, 10 tiles for every 1 inch
    def __init__(self, sizex: float, sizey: float):
        self.sizex = sizex
        self.sizey = sizey
        self.board = [[Tile(self ,i, j, True) for i in range(sizex * TILES_PER_INCH)] for j in range(sizey * TILES_PER_INCH)]
        self.models : list[Model] = []

        

        self.pathfinding_done = True
        self.pathfinding_setup = False

        self.view_port_origin = imgui.ImVec2(0, 0)
        self.view_port_size = imgui.ImVec2(sizex * TILES_PER_INCH, sizey * TILES_PER_INCH)

        # find the right amount of pixels per tile to fit the board in the window
        self.window_size = imgui.ImVec2(800, 600)
        self.pixels_per_tile = min(self.window_size.x / self.view_port_size.x, self.window_size.y / self.view_port_size.y)
        self.ImageRgb = np.zeros((int(sizey * TILES_PER_INCH * self.pixels_per_tile), int(sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)

        self.start_time = 0
        self.tile_time = 0
        self.model_time = 0
        self.pathfinding_time = 0

        self.glfw = None


        

    
    def add_model(self, radius: float, x: int, y: int):
        model = Model(self, radius, x, y)
        self.models.append(model)
    
    def update(self):
        # draw the board as a grid offset from the cursor position
        # find the right amount of pixels per tile to fit the board in the window

        if self.glfw == None:
            self.glfw = glfw_utils.glfw_window_hello_imgui()
        
        
        # board + fps
        imgui.text("FPS: {:.2f}".format(1/imgui.get_io().delta_time))
        # total time
        imgui.text("Total time: {:.4f}".format(time.time() - self.start_time))
        # tile update time 
        imgui.text("Tile update time: {:.4f}".format(self.tile_time - self.start_time))
        # model update time
        imgui.text("Model update time: {:.4f}".format(self.model_time - self.tile_time))
        # pathfinding update time
        imgui.text("Pathfinding update time: {:.4f}".format(self.pathfinding_time - self.model_time))

        self.start_time = time.time()


        # check if the window size has changed and update the pixels per tile
        
        new_window_size = imgui.get_content_region_avail() - imgui.ImVec2(0, 50)

        if new_window_size.x != self.window_size.x or new_window_size.y != self.window_size.y:
            self.window_size = new_window_size
            self.pixels_per_tile = min(self.window_size.x / self.view_port_size.x, self.window_size.y / self.view_port_size.y)
        
        self.ImageRgb = np.zeros((int(self.sizey * TILES_PER_INCH * self.pixels_per_tile), int(self.sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)

        cursor_pos = imgui.get_cursor_pos()
        
        for row in self.board:
            for tile in row:
                tile.update(cursor_pos, self.pixels_per_tile)
        
        immvision.image(
        "##Original", self.ImageRgb, immvision.ImageParams(show_zoom_buttons=False, show_pixel_info=False, show_options_button=False, can_resize=False, refresh_image=True))

        self.tile_time = time.time()


        for model in self.models:
            state, x, y = model.update(cursor_pos, not self.pathfinding_done)

            if state == ModelState.PICKED_UP:
                self.clear_reachable_tiles()
                self.setup_pathfinding(x, y, model.speed, model.radius)
                self.pathfinding_done = False
                break
            elif state == ModelState.PLACED:
                self.clear_reachable_tiles()
                break
        
        self.model_time = time.time()
        
        self.update_pathfinding(cursor_pos)

        self.pathfinding_time = time.time()


        
        # set the cursor position to the row after the end of the board
        imgui.set_cursor_pos(imgui.ImVec2(cursor_pos.x + self.sizex * TILES_PER_INCH * self.pixels_per_tile, cursor_pos.y + self.sizey * TILES_PER_INCH * self.pixels_per_tile))


    def setup_pathfinding(self, x: int, y: int, speed: int, radius: int):
        # a flood fill pathfinding, that gets the neightbors of the current tile and checks if they are reachable and stores the distance from the start tile
        # then the neighbors of the neighbors are checked and so on until the distance is greater than the speed of the model
        self.x = x
        self.y = y
        print(f"setting up pathfinding at {x}, {y}")
        # speed adjusted to radius of the model
        self.speed = speed + radius
        self.open_list = []
        self.board[y][x].is_reachable = True

        #self.open_list.append((x, y, 0))
        # add the neighbors of the start tile to the open list
        for neighbor, distance in self.get_neighbors(x, y):
            if neighbor.is_walkable:
                self.open_list.append((neighbor.x, neighbor.y, distance))

        self.pathfinding_setup = True
        self.pathfinding_done = False
        
    
    def update_pathfinding(self, cursor_pos):
        if not self.pathfinding_done and self.pathfinding_setup:
            start_time = time.time()
            while len(self.open_list) > 0:
                x, y, distance = self.open_list.pop(0)
                if distance < self.speed:
                    self.board[y][x].is_reachable = True
                    for neighbor, neighbor_distance in self.get_neighbors(x, y):
                        if neighbor.is_walkable:
                            # check if the neighbor is reachable and  if so update the distance and readd the neighbor to the open list
                            if neighbor.distance == None:
                                neighbor.distance = distance + neighbor_distance
                                self.open_list.append((neighbor.x, neighbor.y, distance + neighbor_distance))
                            elif distance + neighbor_distance < neighbor.distance:
                                neighbor.distance = distance + neighbor_distance
                                neighbor.edge = False
                                self.open_list.append((neighbor.x, neighbor.y, distance + neighbor_distance))
                else:
                    self.board[y][x].edge = True
                if time.time() - start_time > 0.01:
                    break
            if len(self.open_list) == 0:
                self.pathfinding_done = True    
        if self.pathfinding_setup:
            # draw the speed of the model as a circle around the model
            imgui.get_window_draw_list().add_circle(imgui.ImVec2(cursor_pos.x + self.x * self.pixels_per_tile, cursor_pos.y + self.y * self.pixels_per_tile), self.speed * self.pixels_per_tile, imgui.get_color_u32((0, 0, 255, 255)), num_segments=64)    
    
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


    
    def clear_reachable_tiles(self):
        print("clearing reachable tiles")
        for row in self.board:
            for tile in row:
                tile.is_reachable = False
                tile.distance = None
                tile.edge = False
        self.pathfinding_setup = False
        self.ImageRgb = np.zeros((int(self.sizey * TILES_PER_INCH * self.pixels_per_tile), int(self.sizex * TILES_PER_INCH * self.pixels_per_tile), 3), np.uint8)


# an enum representing the states of a model 
class ModelState:
    IDLE = 0
    MOVING = 1
    PICKED_UP = 2
    PLACED = 3

class Model: 
    def __init__(self, board: Board, radius: float, x: int, y: int, speed: float = 12 ,color: tuple[int, int, int, int] = (255, 0, 0, 255)):
        # x and y are the center position of the model passed in inches, and stored in tiles, radius is the radius of the model passed in inches and stored in tiles
        self.radius = radius * TILES_PER_INCH
        self.x = x * TILES_PER_INCH
        self.y = y * TILES_PER_INCH
        # speed passed in inches, stored in tiles
        self.speed = speed * TILES_PER_INCH
        self.board = board
        self.moving = False
        self.placed = True
        self.last_mouse_pos = imgui.ImVec2(0, 0)
        self.color = color
        self.state = ModelState.IDLE
        self.check_valid_position(check_reachable=False)
        self.current_x = self.x
        self.current_y = self.y
        self.previous_delta = imgui.ImVec2(0, 0)

        
    
    def update(self, cursor_pos, active_pathfinding: bool = False):

        
        # if the mouse is hovering create a tooltip for the model with the x and y position
        if imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile, 
                                                     cursor_pos.y + self.y * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile), 
                                        imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile, 
                                                     cursor_pos.y + self.y * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile)) or self.moving:
            #imgui.begin_tooltip()
            #imgui.text(f"x: {round(self.x, 1)} y: {round(self.y, 1)}")
            #imgui.end_tooltip()
            pass
        elif imgui.is_mouse_clicked(0) and not self.placed:
            self.state = ModelState.PLACED
            self.placed = True
            return (self.state, self.current_x, self.current_y)

        
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

                # update the mouse position to the current position in pixels
                glfw_utils.glfw.set_cursor_pos(self.board.glfw, cursor_pos.x + self.x * self.board.pixels_per_tile, cursor_pos.y + self.y * self.board.pixels_per_tile)
        else:
            if imgui.is_mouse_clicked(0) and imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile, 
                                                                                    cursor_pos.y + self.y * self.board.pixels_per_tile - self.radius * self.board.pixels_per_tile), 
                                                                        imgui.ImVec2(cursor_pos.x + self.x * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile, 
                                                                                    cursor_pos.y + self.y * self.board.pixels_per_tile + self.radius * self.board.pixels_per_tile)):
                
                # set the cursor position to the center of the screen
                imgui.set_mouse_cursor(imgui.MouseCursor_.none)
                screen_center = imgui.ImVec2(cursor_pos.x + 2 * self.board.sizex * self.board.pixels_per_tile, cursor_pos.y + 2 * self.board.sizey * self.board.pixels_per_tile)
                glfw_utils.glfw.set_cursor_pos(self.board.glfw, screen_center.x, screen_center.y)
                self.moving = True
                if self.placed and not active_pathfinding:
                    self.state = ModelState.PICKED_UP
                    print(f"picking up model at {self.x}, {self.y}")
                    self.placed = False
                else:
                    self.state = ModelState.MOVING
            else:
                self.state = ModelState.IDLE



        # convert the x and y position to pixels
        _x = cursor_pos.x   + self.x  * self.board.pixels_per_tile
        _y = cursor_pos.y  + self.y  * self.board.pixels_per_tile
        
        imgui.get_window_draw_list().add_circle_filled(imgui.ImVec2(_x, _y), self.radius * self.board.pixels_per_tile, imgui.get_color_u32((255, 0, 0, 255)), num_segments=64)

        # draw text with the distance of the model from the start tile
        if self.board.board[int(self.y)][int(self.x)].distance != None:
            imgui.get_window_draw_list().add_text(imgui.ImVec2(_x - self.radius * self.board.pixels_per_tile / 2, _y - self.radius * self.board.pixels_per_tile / 2), imgui.get_color_u32((0, 0, 0, 255)), f"{round(self.board.board[int(self.y)][int(self.x)].distance / TILES_PER_INCH, 1)}")

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
        if self.board.board[self.y][self.x].distance != None and self.board.board[self.y][self.x].distance < self.speed:
            pass
        else:
            # if not find the closest reachable tile
            closest_distance = None
            closest_tile = None
            for row in self.board.board:
                for tile in row:
                    if tile.distance != None and tile.distance < self.speed:
                        distance = math.sqrt((self.x - tile.x) ** 2 + (self.y - tile.y) ** 2)
                        if closest_distance == None or distance < closest_distance:
                            closest_distance = distance
                            closest_tile = tile
            if closest_tile != None:
                self.x = closest_tile.x
                self.y = closest_tile.y
            else:
                self.x = self.current_x
                self.y = self.current_y
            
            self.check_valid_position(check_reachable=False)

class Tile:
    def __init__(self, board: Board ,x: int, y: int, is_walkable: bool, is_reachable: bool = False ,color: tuple[int, int, int, int] = (128, 128, 128, 255)):
        self.board = board
        self.x = x
        self.y = y
        self.is_walkable = is_walkable
        self.is_reachable = is_reachable
        self.could_be_reachable = False
        self.edge = False
        self.color = color
        self.distance = None

        self.already_clicked_without_break = False
    
    def update(self, cursor_pos, pixels_per_tile):
        # right click to change the walkable status of the tile
        
        if imgui.is_mouse_down(1) and imgui.is_mouse_hovering_rect(imgui.ImVec2(cursor_pos.x + self.x * pixels_per_tile, cursor_pos.y + self.y * pixels_per_tile), imgui.ImVec2(cursor_pos.x + self.x * pixels_per_tile + pixels_per_tile, cursor_pos.y + self.y * pixels_per_tile + pixels_per_tile)):
            if not self.already_clicked_without_break:
                self.already_clicked_without_break = True
                self.is_walkable = not self.is_walkable
                # change the color of the tile to represent the walkable status
        else:
            self.already_clicked_without_break = False

        
        if self.is_walkable:
            if self.could_be_reachable:
                self.color = (0, 255, 0, 255)
            elif self.is_reachable:
                self.color = (0, 0, 255, 255)
            else:
                self.color = (255, 255, 255, 255)
                return
        else:
            self.color = (255, 0, 0, 255)

        _x = self.x * pixels_per_tile
        _y = self.y * pixels_per_tile

        # draw the tile in board.ImageRgb
        self.board.ImageRgb[int(_y):int(_y + pixels_per_tile), int(_x):int(_x + pixels_per_tile)] = self.color[:3]

        #imgui.get_window_draw_list().add_rect_filled(imgui.ImVec2(_x, _y), imgui.ImVec2(_x + pixels_per_tile, _y + pixels_per_tile), imgui.get_color_u32(self.color))
        #imgui.get_window_draw_list().add_rect(imgui.ImVec2(_x, _y), imgui.ImVec2(_x + pixels_per_tile, _y + pixels_per_tile), imgui.get_color_u32((0, 0, 0, 255)))
        