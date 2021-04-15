# -*- coding: utf-8 -*-
"""
Using Perlin noise to create simple, 'random' shapes
Uses cv2 for canvas and drawing method and noise for perlin noise
"""
from functools import singledispatchmethod
import time

import cv2
import noise
import numpy as np

# Canvas Settings
CANVAS = np.zeros((1000, 1000, 3), np.uint8)

#Helper Function since CV2 doesn't have native alpha channel in images
def rgba2rgb(color, background=(0,0,0), alpha=None):
    """
    Converts RGBA to RGB
    In the case of cv2, greatly increases speed without having to copy background layer

    Keyword Arguments:
        color (tuple): RGBA information. If A not there, will need alpha param defined
        background (tuple): Background color to blend against. Default is black (0, 0, 0)
        alpha (float): If A channel not defined in color tuple, alpha param will cover
    
    Returns:
        tuple: r,g,b values converted as if there was alpha channel
    """
    
    if alpha:
        a = alpha
    else:
        assert len(color) == 4, "Need 4 channel RGBA if no alpha defined"
        a = color[-1]

    r = int(((1-a) * background[0]) + (a*color[0]))
    g = int(((1-a) * background[1]) + (a*color[1]))
    b = int(((1-a) * background[2]) + (a*color[2]))

    return (r,g,b)

class PLine:
    
    """
    Perlin Line Object that moves accordingly to Noise
    
    Keyword Arguments:
        center (tuple: int): Center location for PLine to start
    """
    
    def __init__(self, center):

        self.x, self.y = center
        self.xi, self.yi = center # Position before movement
        self.xo, self.yo = center # Original X, Y. Should never change

        # Create random movement speads in the PLine
        self.xinc = np.random.randint(-10, 10)
        self.yinc = np.random.randint(-10, 10)
        
        # Random Z dimension in 3d Perlin Noise
        self.z = np.random.randint(-10, 10)

    def timedelta(self):
        """ Adds time based change in Z dimension """
        return time.time()*0.0001 - self.z

    def perlin_move(self):
        """ Draw and update position of Perlin Line """

        px = noise.pnoise3(self.x*0.01, self.y*0.01, self.timedelta())
        py = noise.pnoise3(self.y*0.01, self.x*0.01, self.timedelta())

        self.xinc += px
        self.yinc += py
        
        self.xi = self.x
        self.yi = self.y
        
        self.x += self.xinc
        self.y += self.yinc
        
    def __call__(self):
        """ Alternative method for running perlin_move """
        self.perlin_move()

# Setup Grid of Perlin Lines

class PShape:
    """ 
    Manage collective of Perlin Lines 
    
    Keyword Arguments:
        center (tuple: int): Center location for PLine to start
        xbounds (tuple: int): X-Axis boundaries in which Perlin can move within
        ybounds (tuple: int): Y-Axis boundaries in which Perlin can move within
        lines (int): Generates number of Perlin Lines
        shape (str): Shape that will contain the Perlin Noise
    """
    
    def __init__(self, center, xbounds, ybounds, lines, shape="rectangle"):
        
        self.plines = [PLine((center[0], center[1])) for _ in range(lines)]
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.shape = shape
        
    @singledispatchmethod
    def within(self):
        """ 
        Determine if position or pline is in given constraints 
        Uses singledispatch method to handle both cases
        """
        raise NotImplementedError("Within only supports np.float64 or Pline Object")
    
    @within.register(np.float64)
    def _(self, position, bounds):
        """
        np.float64 handling of within. Used to check if position in CANVAS
        
        Keyword Arguments:
            position (int): 1D position
            bounds (tuple: int): 1D Bounding Range
            
        Returns:
            boolean: True if within bounds, else False
        """
        within_bool = False
        
        if position >= bounds[0] and position <= bounds[1] :
            within_bool = True
        
        return within_bool
    
    @within.register(PLine)
    def _(self, pline):
        """
        Pline handling of within. Used for shape within checking
        
        Keyword Arguments:
            pline (Pline): Perlin line object
            
        Returns:
            boolean: True if within bounds, else False
        """
        
        within_bool = False
        
        if self.shape == "rectangle":
            if pline.xi >= self.xbounds[0] and pline.xi <= self.xbounds[1]:
                if pline.yi >= self.ybounds[0] and pline.yi <= self.ybounds[1]:
                    within_bool = True
        elif self.shape == "circle":
            if ((pline.xi-pline.xo)**2)/((self.xbounds[1]-self.xbounds[0])**2) + \
                ((pline.yi-pline.yo)**2)/((self.ybounds[1]-self.ybounds[0])**2) <= 1:
                within_bool = True
    
        return within_bool
    
    def perlin_move(self, pline):
        
        if self.within(pline.xi, (0, CANVAS.shape[0])) and self.within(pline.yi, (0, CANVAS.shape[1])):
            if self.within(pline):
                cv2.line(CANVAS,
                         (np.float32(pline.xi), np.float32(pline.yi)),
                         (np.float32(pline.x), np.float32(pline.y)),
                         rgba2rgb((255, 255, 255, 0.2), [int(_) for _ in CANVAS[int(pline.yi)][int(pline.xi)]]),
                         1)
        
    def __call__(self):
        """ Call on PSquare iterates through all PLines and runs perlin_move """
        for pline in self.plines:
            self.perlin_move(pline)
            pline()

def grid(n=None, rows=None, columns=None, margin=None, 
         xbounds=100, ybounds=100, shape="rectangle"):
    """ 
    Handles creating the grid of Perlin Squares
    
    Keyword Arguments;
        b (int): Creates n^2 squares in grid
        rows (int): Rows of Squares
        columns (int): Columns of Squares
        margin (int) Spacing between edges of squares
        xbounds (tuple: int): X-Axis boundaries in which Perlin can move within
        ybounds (tuple: int): Y-Axis boundaries in which Perlin can move within
        shape (str): Shape that will contain the Perlin Noise
        
    Returns:
        list: Perlin Squares
    """
    
    if n:
        xspace = np.linspace(0+margin, CANVAS.shape[0]-margin, n)
        yspace = np.linspace(0+margin, CANVAS.shape[1]-margin, n)
    else:
        assert rows and columns, f"Rows and Columns must be defined if not using n.  Rows: {rows} Columns {columns}"
        xspace = np.linspace(0+margin, CANVAS.shape[0]-margin, columns)
        yspace = np.linspace(0+margin, CANVAS.shape[1]-margin, rows)
    
    grid =[]
    for x in xspace:
      for y in yspace:
          grid.extend([PShape((x, y), (x-xbounds, x+xbounds), (y-ybounds, y+ybounds), np.random.randint(20,80), shape=shape)])
    return grid
        

#Create Here!
psquares = grid(n=3, margin=200) # Perlin Square information to display
delay = 10 # Delay in Milliseconds to draw
capture = True # Record to 'output.avi'

if capture:
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    out = cv2.VideoWriter('output.avi', fourcc, delay, (1000, 1000))

while True:
    _CANVAS = CANVAS.copy()
    # Run all movements on all polygons
    for psquare in psquares:
        psquare()
        
    # Another alpha tranparency measure
    CANVAS = cv2.addWeighted(CANVAS, 0.5, _CANVAS, 0.5, 0)
    cv2.imshow("Canvas", CANVAS)
    if capture:
        out.write(CANVAS)

    # Exit CANVAS by clicking 'q' on your keyboard
    if cv2.waitKey(delay) & 0xFF == ord('q'): 
        break

if capture:
    out.release()
cv2.destroyAllWindows()

