"""
简化版数据提取工具
不依赖opencv，适用于基本的数据提取需求

使用步骤：
1. 将脚本放在包含图片的文件夹中运行。
2. 选择要加载的图片文件。
3. 点击图片标记感兴趣的区域。
4. 调整右侧滑块选择颜色类别。
5. 点击“保存数据”按钮保存标记结果。
6. 点击“清除所有”按钮清除所有标记。
7. 点击“撤销”按钮撤销上一步操作。
"""

import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import json
from PIL import Image
import os
from datetime import datetime

# 过滤字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleDataExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.fig = None
        self.ax = None
        self.points = []
        self.current_color_class = 1
        
        
        # 预定义颜色类别
        self.color_classes = {
            1: {'name': '0-10%', 'color': 'white'},
            2: {'name': '10-20%', 'color': 'lightgray'},
            3: {'name': '20-30%', 'color': 'gray'},
            4: {'name': '30-40%', 'color': 'darkgray'},
            5: {'name': '40-50%', 'color': 'lightblue'},
            6: {'name': '50-60%', 'color': 'blue'},
            7: {'name': '60-70%', 'color': 'green'},
            8: {'name': '70-80%', 'color': 'yellow'},
            9: {'name': '80-90%', 'color': 'orange'},
            10: {'name': '90-100%', 'color': 'red'}
        }
        
        self.load_image()
        self.setup_interface()
    
    def load_image(self):
        """加载图片"""
        try:
            self.image = Image.open(self.image_path)
            self.image_array = np.array(self.image)
            print(f"成功加载图片: {self.image_path}")
            print(f"图片尺寸: {self.image.size}")
        except Exception as e:
            print(f"加载图片失败: {e}")
    
    def geo_to_pixel(self, lon, lat):
        """将经纬度坐标转换为像素坐标"""
        if self.image is None:
            return None, None
        
        # 获取图片尺寸
        img_width, img_height = self.image.size
        
        # 转换为像素坐标
        pixel_x = ((lon - self.geo_bounds['min_lon']) / 
                  (self.geo_bounds['max_lon'] - self.geo_bounds['min_lon'])) * img_width
        
        pixel_y = ((self.geo_bounds['max_lat'] - lat) / 
                  (self.geo_bounds['max_lat'] - self.geo_bounds['min_lat'])) * img_height
        
        # 应用校正参数
        corrected_x = pixel_x / self.geo_calibration['scale_x'] + self.geo_calibration['offset_x']
        corrected_y = pixel_y / self.geo_calibration['scale_y'] + self.geo_calibration['offset_y']
        
        return corrected_x, corrected_y
    
    def setup_interface(self):
        """设置交互界面"""
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15, right=0.8)
        
        # 显示底图
        self.ax.imshow(self.image_array)
        self.ax.set_title('数据提取工具 - 点击图片标记散点')
        
        # 添加控制区域
        self.add_controls()
        
        # 绑定事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 显示使用说明
        self.show_instructions()
        
        plt.show()
    
    def add_controls(self):
        """添加控制按钮"""
        # 颜色类别选择
        ax_color = plt.axes([0.82, 0.85, 0.15, 0.03])
        self.slider_color = Slider(ax_color, '类别', 1, 10, valinit=1, valfmt='%d')
        self.slider_color.on_changed(self.update_color_class)
        
        # 保存按钮
        ax_save = plt.axes([0.82, 0.6, 0.15, 0.04])
        self.btn_save = Button(ax_save, '保存数据')
        self.btn_save.on_clicked(self.save_data)
        
        # 清除按钮
        ax_clear = plt.axes([0.82, 0.55, 0.15, 0.04])
        self.btn_clear = Button(ax_clear, '清除所有')
        self.btn_clear.on_clicked(self.clear_points)
        
        # 撤销按钮
        ax_undo = plt.axes([0.82, 0.5, 0.15, 0.04])
        self.btn_undo = Button(ax_undo, '撤销')
        self.btn_undo.on_clicked(self.undo_last_point)
        
        # 显示当前状态
        self.update_status_display()
    
    def show_instructions(self):
        """显示使用说明"""
        instructions = """使用说明:
左键: 标记点
右键: 删除最近点
数字键: 切换类别
S: 保存  C: 清除  Z: 撤销"""
        
        self.ax.text(0.02, 0.02, instructions, transform=self.ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9, verticalalignment='bottom')
    
    def update_color_class(self, val):
        """更新颜色类别"""
        self.current_color_class = int(self.slider_color.val)
        self.update_status_display()
    
    def update_status_display(self):
        """更新状态显示"""
        class_info = self.color_classes[self.current_color_class]
        status_text = f"当前类别: {self.current_color_class}\n"
        status_text += f"名称: {class_info['name']}\n"
        status_text += f"已标记: {len(self.points)}个点"
        
        if hasattr(self, 'status_text'):
            self.status_text.remove()
        
        self.status_text = self.ax.text(0.98, 0.98, status_text,
                                      transform=self.ax.transAxes,
                                      verticalalignment='top',
                                      horizontalalignment='right',
                                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        if hasattr(self, 'fig'):
            self.fig.canvas.draw()
    
    def on_click(self, event):
        """处理鼠标点击"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if event.button == 1:  # 左键 - 添加点
            self.add_point(x, y)
        elif event.button == 3:  # 右键 - 删除最近点
            self.remove_nearest_point(x, y)
    
    def add_point(self, x, y):
        """添加点"""
        # 获取像素颜色
        pixel_color = self.get_pixel_color(int(x), int(y))
        
        # 创建点信息
        point_info = {
            'id': len(self.points) + 1,
            'x': x,
            'y': y,
            'pixel_color': pixel_color,
            'assigned_class': self.current_color_class,
            'class_name': self.color_classes[self.current_color_class]['name'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.points.append(point_info)
        
        # 在图上绘制点
        self.draw_point(x, y, self.current_color_class, len(self.points))
        
        print(f"添加点 {len(self.points)}: ({x:.1f}, {y:.1f}) - {point_info['class_name']}")
        self.update_status_display()
    
    def remove_nearest_point(self, x, y):
        """删除最近的点"""
        if not self.points:
            return
        
        # 找到最近的点
        min_distance = float('inf')
        nearest_index = -1
        
        for i, point in enumerate(self.points):
            dist = ((point['x'] - x) ** 2 + (point['y'] - y) ** 2) ** 0.5
            if dist < min_distance:
                min_distance = dist
                nearest_index = i
        
        # 如果距离小于20像素则删除
        if nearest_index >= 0 and min_distance < 20:
            removed = self.points.pop(nearest_index)
            print(f"删除点: ({removed['x']:.1f}, {removed['y']:.1f})")
            
            # 重新编号和重绘
            for i, point in enumerate(self.points):
                point['id'] = i + 1
            
            self.redraw_all_points()
            self.update_status_display()
    
    def get_pixel_color(self, x, y):
        """获取像素颜色"""
        try:
            if 0 <= x < self.image_array.shape[1] and 0 <= y < self.image_array.shape[0]:
                return tuple(self.image_array[y, x][:3])  # RGB
            return (0, 0, 0)
        except:
            return (0, 0, 0)
    
    def draw_point(self, x, y, color_class, point_id):
        """绘制单个点"""
        class_info = self.color_classes[color_class]
        color = class_info['color']
        
        # 绘制空心圆圈（只显示轮廓，不填充颜色）
        self.ax.plot(x, y, 'o', markerfacecolor='none', 
                    markeredgecolor=color, markeredgewidth=2, 
                    markersize=8, alpha=0.9)
        
        self.fig.canvas.draw()
    
    def redraw_all_points(self):
        """重新绘制所有点"""
        # 清除图像并重新显示
        self.ax.clear()
        self.ax.imshow(self.image_array)
        self.ax.set_title('数据提取工具 - 点击图片标记散点')
        
        # 重新绘制所有点
        for point in self.points:
            self.draw_point(point['x'], point['y'], point['assigned_class'], point['id'])
        
        # 重新显示说明
        self.show_instructions()
        self.update_status_display()
    
    def clear_points(self, event):
        """清除所有点"""
        self.points = []
        self.redraw_all_points()
        print("已清除所有点")
    
    def undo_last_point(self, event):
        """撤销最后一个点"""
        if self.points:
            removed = self.points.pop()
            print(f"撤销点: ({removed['x']:.1f}, {removed['y']:.1f})")
            
            # 重新编号
            for i, point in enumerate(self.points):
                point['id'] = i + 1
            
            self.redraw_all_points()
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key.isdigit():
            class_num = int(event.key)
            if 1 <= class_num <= 10:
                self.current_color_class = class_num
                self.slider_color.set_val(class_num)
                self.update_status_display()
        elif event.key == 's':
            self.save_data(None)
        elif event.key == 'c':
            self.clear_points(None)
        elif event.key == 'z':
            self.undo_last_point(None)
    
    def save_data(self, event):
        """保存数据"""
        if not self.points:
            print("没有数据可保存")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存CSV
        df = pd.DataFrame(self.points)
        csv_file = f"data_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print(f"数据已保存:")
        print(f"  CSV: {csv_file}")
        print(f"  总点数: {len(self.points)}")
        
        # 统计各类别
        stats = {}
        for point in self.points:
            class_name = point['class_name']
            stats[class_name] = stats.get(class_name, 0) + 1
        
        print("类别统计:")
        for class_name, count in sorted(stats.items()):
            print(f"  {class_name}: {count}个")

def main():
    """主函数"""
    print("汶川地震数据提取工具（简化版）")
    print("=" * 40)
    
    # 查找图片文件
    image_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)
    
    if not image_files:
        print("未找到图片文件")
        return
    
    print("可用图片:")
    for i, file in enumerate(image_files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input(f"选择图片 (1-{len(image_files)}): ")) - 1
        if 0 <= choice < len(image_files):
            selected = image_files[choice]
            print(f"加载: {selected}")
            SimpleDataExtractor(selected)
        else:
            print("选择无效")
    except (ValueError, KeyboardInterrupt):
        print("程序退出")

if __name__ == "__main__":
    main()
