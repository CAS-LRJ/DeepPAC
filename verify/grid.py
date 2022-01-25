import math

'''
    Classes:
        Grid: The Grids Used For Stepwise Splitting
    
    Functions:
        grid_split: Split a List of Grid Objects into 2x2 sub-grids.
'''


class Grid(object):

    def __init__(self, leftup_x, leftup_y, rightdown_x, rightdown_y) -> None:
        self.leftup_x = leftup_x
        self.leftup_y = leftup_y
        self.rightdown_x = rightdown_x
        self.rightdown_y = rightdown_y

    def __str__(self) -> str:
        return 'Grid: (%d,%d) -> (%d,%d)' % (self.leftup_x, self.leftup_y, self.rightdown_x, self.rightdown_y)


def grid_split(grid_list, x_split, y_split):
    new_grid_list = []
    for grid in grid_list:
        x_step = math.ceil((grid.rightdown_x-grid.leftup_x)/x_split)
        y_step = math.ceil((grid.rightdown_y-grid.leftup_y)/y_split)
        for x in range(grid.leftup_x, grid.rightdown_x, x_step):
            for y in range(grid.leftup_y, grid.rightdown_y, y_step):
                x_ = min(x+x_step, grid.rightdown_x)
                y_ = min(y+y_step, grid.rightdown_y)
                new_grid_list.append(Grid(x, y, x_, y_))
    return new_grid_list
