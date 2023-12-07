from PIL import Image
import cv2
import numpy as np


from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    invocation,
    invocation_output,
    OutputField,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
)

from invokeai.app.invocations.primitives import (
    ImageField,
    ColorField,
    ColorOutput
)


@invocation_output("simple_skin_detection_output")
class SimpleSkinDetectionOutput(BaseInvocationOutput):
    mask: ImageField = OutputField(description="The output image general mask")
    mask_hsv: ImageField = OutputField(description="The output image HSV mask")
    mask_ycrcb: ImageField = OutputField(description="The output image YCbCr mask")
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")
    mediant_skin_color: ColorField = OutputField(description="Mediant skin color")
    max_skin_color: ColorField = OutputField(description="Maximum skin color")



@invocation(
    "simple_skin_detection",
    title="Simple Skin Detection",
    tags=["image", "skin"],
    category="image",
    version="1.0.0",
)
class SimpleSkinDetectionInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Skin and skin tone detection with HSV and YCbCr"""
    image: ImageField = InputField(default=None, description="Input image")
    iterations: int = InputField(default=1, description="The number of times morphological operations will be applied")
    kernel_size: int = InputField(default=3, description="The size of the kernel for morphological operations")
    smooth_outline: int = InputField(default=3, description="The amount of blurring applied to the image's edges for a smoother outline")


    def invoke(self, context: InvocationContext) -> SimpleSkinDetectionOutput:
        image = context.services.images.get_pil_image(self.image.image_name)  
        cv_image = self.pil2cv2_image(image)
        
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(image_hsv, (0, 15, 0), (17, 170, 255)) 
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=self.iterations)
        
        image_ycrcb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(image_ycrcb, (0, 135, 85), (255, 180, 135)) 
        mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=self.iterations)

        mask_global = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        mask_global = cv2.medianBlur(mask_global, self.smooth_outline)
        mask_global = cv2.morphologyEx(mask_global, cv2.MORPH_OPEN, np.ones((self.kernel_size + 1, self.kernel_size + 1), np.uint8), iterations=self.iterations)

        colors = self.get_skin_color(cv_image)

        result_hsv_dto = context.services.images.create(
            image=self.cv2Pilimage(mask_hsv),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        result_ycrcb_dto = context.services.images.create(
            image=self.cv2Pilimage(mask_ycrcb),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        result_global_dto = context.services.images.create(
            image=self.cv2Pilimage(mask_global),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return SimpleSkinDetectionOutput(
            mask=ImageField(image_name=result_global_dto.image_name),
            mask_hsv=ImageField(image_name=result_hsv_dto.image_name),
            mask_ycrcb=ImageField(image_name=result_ycrcb_dto.image_name),
            width=result_global_dto.width,
            height=result_global_dto.height,
            mediant_skin_color=ColorField(r=colors[0][0], g=colors[0][1], b=colors[0][2], a=255),
            max_skin_color=ColorField(r=colors[1][0], g=colors[1][1], b=colors[1][2], a=255)
        )
    

    def pil2cv2_image(self, image):
        numpy_image = np.array(image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
    

    def cv2Pilimage(self, image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    

    def get_skin_color(self, cv_image):
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, (0, 15, 0), (17, 170, 255))
        skin = cv2.bitwise_and(cv_image, cv_image, mask=mask)    
        rows, cols, channels  = skin.shape
        arr = np.array([0, 0, 0])

        blue = [] 
        green = [] 
        red = [] 
        for i in range(rows):
            for j in range(cols):
                k = skin[i, j]
                if not (k[0] == arr[0] and k[1] == arr[1] and k[2] == arr[2]):
                    blue.append(k[0])
                    green.append(k[1])
                    red.append(k[2])
           
        return [
            [   # Mediant color
                int((sum(red) / len(red))),
                int((sum(green) / len(green))),
                int((sum(blue) / len(blue)))
            ],
            [   # Max Color
                int(max(red)),
                int(max(green)),
                int(max(blue))
            ]
        ]