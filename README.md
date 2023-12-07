# Simple Skin Detection

<img src=".readme/node.png" style="float: left;" />

The "Simple Skin Detection" node is a rudimentary approach to identifying skin in images by applying color thresholding in both the HSV and YCrCb color spaces, as human skin tends to fall within a specific color range in these spaces. This node generates a mask that attempts to capture areas of an image that resemble these skin tones.

While this straightforward method provides a basic level of skin detection, it is not without its drawbacks. The simplicity of the algorithm can lead to false positives, such as detecting light-colored clothing or other objects with colors similar to human skin. This can result in a mask that includes more than just skin areas, capturing unintended parts of the image.
