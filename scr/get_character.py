import image_analysis
from typing import TypedDict, List


class ImageItem(TypedDict):
    name: str  # 图片名称
    img_path: str  # 图片路径
    description: str  # 图片描述


class Character:
    def __init__(self, name: str, age: int, gender: str, image: list[ImageItem]):
        """
        :param image: 本地图片的路径列表[{'name':'','img_path':'', 'description':''}]
        description默认为空， get_img_analysis（）得到的内容将追加到description中。即初始人为描述加ai描述
        """

        self.name = name
        self.age = age
        self.gender = gender
        self.image = image

        # 默认使用siliconflow的qwen模型

    def get_img_analysis(self):
        for element in self.image:
            analysis = image_analysis.analyse_image(element['img_path'])
            element['description'] += f'额外描述：{analysis}'
