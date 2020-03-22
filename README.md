# canvas_character_recognition

Track a color, and draw characters on the screen through a webcam, allow the characters to be recognized and displayed on the screen.

![](char_recog.gif)

### How to use it
* Choose a color you want to track, then press enter
* If it's tracking smoothly and correctly, draw with the spacebar
* Change the color to track by pressing 'r', clicking a new color, and pressing enter
* Erase all the drawings by pressing 'e'
* 'Up' to make cursor bigger
* 'Down' to make cursor smaller
* To quit press 'q'


### How it works
Clicking a color gives the color in rgb and hsv colorspace, a color slightly below and above that color in hsv space is chosen.
Every new frame, all the colors between these values are picked up, this is the 'mask'. The mask is eroded to get rid of 
small pixels that mistakenly were picked up, and clean some of the edges around everything. Then the mask is dilated, to inflate the correct 
areas that are within that color range, this fills in all the gaps that were eroded from the previous step, and then adds some extra padding 
to the correct region.
The area in the mask that has the largest contour is then considered the object to track. 

Characters are separated from the canvas by the space between them. If they are touching, it will consider them as one and not work correctly. Only correctly separates horizontally, not vertically, so only one line of characters will be separated at this moment. 

Trained neural network on the emnist balanced dataset: https://arxiv.org/abs/1702.05373 

