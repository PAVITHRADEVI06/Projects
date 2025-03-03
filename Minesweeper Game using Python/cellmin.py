import sys
from tkinter import Button, Label
import random
import settingsmin
import ctypes

class cell:
    all = []
    cell_count = settingsmin.CELL_COUNT
    cell_count_label_object = None
    #the parameters/arguments are the attributes of a function or class
    #then to call the attributes within the method(def __init__) itself where it is defined you need to use self
    def __init__(self, x, y, is_mine=False):  # constructor is going to be called immediately after a class is initiated
        self.is_mine = is_mine  # self is to use the instances in a class and __init__ is used to initiate constructor
        self.cell_btn_object = None
        self.is_open = False
        self.is_mine_candidate = False
        self.x = x
        self.y = y

        # append the object to the cell.all list
        cell.all.append(self)
#there is a for loop in mines.py(almost at the end) helps us determine the x
    def create_btn_object(self, location):
        btn = Button(
            location,
            width=12,
            height=4,
        )
        btn.bind('<Button-1>', self.left_click_actions)  # with bind we say we'd like to print something when we left/right click on a button
        btn.bind('<Button-3>', self.right_click_actions)
        self.cell_btn_object = btn
    @staticmethod
    def create_cell_count_label(location):
        lbl = Label(
            location,
            bg='black',
            fg='white',
            font=("", 30),
            text=f"Cells Left:{cell.cell_count}"
        )
        cell.cell_count_label_object = lbl
    def left_click_actions(self, event):
        if self.is_mine:
            self.show_mine()
        else:
            if self.surrounding_cells_mines_length == 0:
                for cell_obj in self.surrounding_cells:
                    cell_obj.show_cell()
            self.show_cell()
            #if mines count is equal to the cells left count, player won
            if cell.cell_count == settingsmin.MINES_COUNT:
                ctypes.windll.user32.MessageBoxW(0, 'You Won', 'Game Over', 0)
                sys.exit()
    #cancel left and right click events if cell is already opened
        self.cell_btn_object.unbind('<Button-1>')
        self.cell_btn_object.unbind('<Button-3>')
    def get_cell_by_axis(self, x,y):
        # return a cell object(that exact particular button) based on the value of x,y
        # cell.all, saves all the buttons generated
        for cells in cell.all:
            if cells.x == x and cells.y == y:
                return cells
    @property
    def surrounding_cells(self):
        Cells = [
            self.get_cell_by_axis(self.x - 1, self.y - 1),
            self.get_cell_by_axis(self.x - 1, self.y),
            self.get_cell_by_axis(self.x - 1, self.y + 1),
            self.get_cell_by_axis(self.x, self.y - 1),
            self.get_cell_by_axis(self.x + 1, self.y - 1),
            self.get_cell_by_axis(self.x + 1, self.y),
            self.get_cell_by_axis(self.x + 1, self.y + 1),
            self.get_cell_by_axis(self.x, self.y + 1)
        ]
        Cells = [cell for cell in Cells if cell is not None]
        return Cells
    #counts the mines in the surrounding cells
    @property
    def surrounding_cells_mines_length(self):
        counter = 0
        for cell in self.surrounding_cells:
            if cell.is_mine:
                counter += 1
        return counter
    def show_cell(self):
        if not self.is_open:
            cell.cell_count -= 1
            self.cell_btn_object.configure(text=self.surrounding_cells_mines_length)
            # replace the text of cell count label with the newer count
            if cell.cell_count_label_object:
                cell.cell_count_label_object.configure(text=f"Cells Left:{cell.cell_count}")
            #if this was a mine candidate, then for safety, we should configure the bg to systembuttonface
        self.cell_btn_object.configure(bg='SystemButtonFace')
        #mark the cell as opened (use it as the last line of this method)
        self.is_open = True
    def show_mine(self):
        # a logic to interrupt the game and display a message that player lost!
        self.cell_btn_object.configure(bg='red')
        ctypes.windll.user32.MessageBoxW(0, 'You Clicked On a Mine', 'Game Over', 0)
        sys.exit()
    def right_click_actions(self, event):
        if not self.is_mine_candidate:
            self.cell_btn_object.configure(
                bg="orange"
            )
            self.is_mine_candidate = True
        else:
            self.cell_btn_object.configure(bg='SystemButtonFace')
            self.is_mine_candidate = False

    # 9 mines technically by MINES_COUNT
    @staticmethod
    def randomize_mines():
        picked_cells = random.sample(
            cell.all, settingsmin.MINES_COUNT
        )
        for picked_cell in picked_cells:
            picked_cell.is_mine = True

    def __repr__(self):  # changes the way object is being represented
        return f"cell({self.x},{self.y})"

# instances are individual object of a particular class
# instantiation creating an instance of a class
# instance method a special kind of function that is defined in a class definition
# is_mine is used to determine whether a given button is a mine or not
# anything inside def() is class definition and self.instance is assigning a value to the variables declared in a function definition
# static to commonly use instances globally in class at any functions in a class