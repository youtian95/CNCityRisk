"""
像素坐标转经纬度坐标转换工具
用于将从图片中提取的像素坐标转换为实际的经纬度坐标

工作流程：
1. 加载地图图像
2. 用户在图像上选择已知地理位置的控制点
3. 输入这些控制点的真实经纬度坐标
4. 建立像素坐标到地理坐标的转换关系
5. 转换数据文件中的所有像素坐标

适用于汶川地震房屋倒塌率分布图等地理图像
"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import Button, TextBox
import os
import sys
from scipy.spatial import distance
from scipy.interpolate import griddata

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CoordinateConverter:
    def __init__(self, image_path):
        """
        初始化坐标转换器
        
        参数:
        image_path: 原始图片路径
        """
        self.image_path = image_path
        self.image = None
        self.image_array = None
        self.img_width = 0
        self.img_height = 0
        
        # 控制点数据：存储像素坐标和对应的经纬度
        self.control_points = []  # [{pixel_x, pixel_y, lon, lat, name}]
        
        # 界面相关
        self.fig = None
        self.ax = None
        self.current_point_name = ""
        self.pending_point = None  # 存储待输入坐标的点
        
        # 输入框组件
        self.textbox_name = None
        self.textbox_lon = None
        self.textbox_lat = None
        
        self.load_image()
    
    def load_image(self):
        """加载图片并获取尺寸信息"""
        try:
            self.image = Image.open(self.image_path)
            self.image_array = np.array(self.image)
            self.img_width, self.img_height = self.image.size
            print(f"成功加载图片: {self.image_path}")
            print(f"图片尺寸: {self.img_width} x {self.img_height}")
            return True
        except Exception as e:
            print(f"加载图片失败: {e}")
            return False
    
    def setup_calibration_interface(self):
        """设置校准界面，让用户选择控制点"""
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.25, right=0.7)
        
        # 显示图像
        self.ax.imshow(self.image_array)
        self.ax.set_title('地理坐标校准 - 请在已知位置点击并在右侧输入经纬度')
        
        # 添加控制按钮和输入框
        self.add_calibration_controls()
        
        # 绑定鼠标点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_calibration_click)
        
        # 显示使用说明
        self.show_calibration_instructions()
        
        plt.show()
    
    def add_calibration_controls(self):
        """添加校准控制按钮和输入框"""
        # 输入区域标题
        self.ax.text(0.72, 0.95, '控制点信息输入', transform=self.fig.transFigure,
                    fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 地点名称输入框
        ax_name_label = plt.axes([0.72, 0.88, 0.25, 0.03])
        ax_name_label.text(0, 0.5, '地点名称:', fontsize=10, verticalalignment='center')
        ax_name_label.set_xlim(0, 1)
        ax_name_label.set_ylim(0, 1)
        ax_name_label.axis('off')
        
        ax_name = plt.axes([0.72, 0.85, 0.25, 0.03])
        self.textbox_name = TextBox(ax_name, '', initial='')
        
        # 经度输入框
        ax_lon_label = plt.axes([0.72, 0.80, 0.25, 0.03])
        ax_lon_label.text(0, 0.5, '经度 (如: 103.5892):', fontsize=10, verticalalignment='center')
        ax_lon_label.set_xlim(0, 1)
        ax_lon_label.set_ylim(0, 1)
        ax_lon_label.axis('off')
        
        ax_lon = plt.axes([0.72, 0.77, 0.25, 0.03])
        self.textbox_lon = TextBox(ax_lon, '', initial='')
        
        # 纬度输入框
        ax_lat_label = plt.axes([0.72, 0.72, 0.25, 0.03])
        ax_lat_label.text(0, 0.5, '纬度 (如: 31.4759):', fontsize=10, verticalalignment='center')
        ax_lat_label.set_xlim(0, 1)
        ax_lat_label.set_ylim(0, 1)
        ax_lat_label.axis('off')
        
        ax_lat = plt.axes([0.72, 0.69, 0.25, 0.03])
        self.textbox_lat = TextBox(ax_lat, '', initial='')
        
        # 确认添加按钮
        ax_confirm = plt.axes([0.72, 0.64, 0.12, 0.04])
        self.btn_confirm = Button(ax_confirm, '确认添加')
        self.btn_confirm.on_clicked(self.confirm_add_point)
        
        # 取消按钮
        ax_cancel = plt.axes([0.85, 0.64, 0.12, 0.04])
        self.btn_cancel = Button(ax_cancel, '取消')
        self.btn_cancel.on_clicked(self.cancel_add_point)
        
        # 控制按钮区域
        self.ax.text(0.72, 0.58, '控制操作', transform=self.fig.transFigure,
                    fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 完成校准按钮
        ax_finish = plt.axes([0.72, 0.52, 0.25, 0.04])
        self.btn_finish = Button(ax_finish, '完成校准')
        self.btn_finish.on_clicked(self.finish_calibration)
        
        # 清除所有按钮
        ax_clear = plt.axes([0.72, 0.47, 0.25, 0.04])
        self.btn_clear = Button(ax_clear, '清除所有点')
        self.btn_clear.on_clicked(self.clear_control_points)
        
        # 删除最后点按钮
        ax_remove = plt.axes([0.72, 0.42, 0.25, 0.04])
        self.btn_remove = Button(ax_remove, '删除最后点')
        self.btn_remove.on_clicked(self.remove_last_point)
        
        # 状态显示
        self.update_calibration_status()
        
        # 初始状态：输入框不可用
        self.set_input_enabled(False)
    
    def show_calibration_instructions(self):
        """显示校准使用说明"""
        instructions = """校准说明:
1. 点击地图上已知地理位置的点
2. 在右侧输入框中填写该点信息
3. 点击"确认添加"保存控制点
4. 重复至少3个点以建立转换关系
5. 点击"完成校准"进行坐标转换

建议选择的控制点：
- 县城中心、乡镇中心
- 主要河流交汇点  
- 山峰等明确地标
- 公路交叉口"""
        
        self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                    fontsize=9, verticalalignment='top')
    
    def update_calibration_status(self):
        """更新校准状态显示"""
        status_text = f"已标记控制点: {len(self.control_points)}个\n"
        status_text += f"最少需要: 3个点\n"
        status_text += f"推荐: 4-6个点"
        
        if hasattr(self, 'status_text'):
            self.status_text.remove()
        
        self.status_text = self.ax.text(0.72, 0.35, status_text,
                                      transform=self.fig.transFigure,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # 显示已有控制点列表
        if self.control_points:
            points_text = "控制点列表:\n"
            for i, point in enumerate(self.control_points, 1):
                points_text += f"{i}. {point['name'][:12]}\n"
                points_text += f"   像素: ({point['pixel_x']:.0f}, {point['pixel_y']:.0f})\n"
                points_text += f"   坐标: ({point['lon']:.5f}, {point['lat']:.5f})\n"
            
            if hasattr(self, 'points_text'):
                self.points_text.remove()
            
            self.points_text = self.ax.text(0.72, 0.28, points_text,
                                          transform=self.fig.transFigure,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                                          fontsize=8)
        
        self.fig.canvas.draw()
    
    def set_input_enabled(self, enabled):
        """设置输入框是否可用"""
        if self.textbox_name:
            # 清空输入框
            self.textbox_name.set_val('')
            self.textbox_lon.set_val('')
            self.textbox_lat.set_val('')
            
            # 设置颜色表示状态
            color = 'white' if enabled else 'lightgray'
            self.textbox_name.ax.set_facecolor(color)
            self.textbox_lon.ax.set_facecolor(color)
            self.textbox_lat.ax.set_facecolor(color)
            
        # 设置按钮状态
        if hasattr(self, 'btn_confirm'):
            self.btn_confirm.ax.set_facecolor('lightgreen' if enabled else 'lightgray')
            self.btn_cancel.ax.set_facecolor('lightcoral' if enabled else 'lightgray')
        
        self.fig.canvas.draw()
    
    def on_calibration_click(self, event):
        """处理校准模式下的鼠标点击"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if event.button == 1:  # 左键点击
            if self.pending_point is None:  # 没有待处理的点
                self.pending_point = {'pixel_x': x, 'pixel_y': y}
                
                # 在图上标记临时点
                self.clear_temp_markers()  # 先清除之前的临时标记
                
                self.temp_point = self.ax.plot(x, y, 'yo', markersize=12, 
                                             markeredgecolor='red', markeredgewidth=2, 
                                             alpha=0.8, label='待输入坐标')[0]
                
                # 添加提示文字
                self.temp_text = self.ax.text(x+10, y-10, '请在右侧输入该点信息', fontsize=9, 
                                            color='red', weight='bold',
                                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                # 激活输入框
                self.set_input_enabled(True)
                
                self.fig.canvas.draw()
                print(f"已选择点: 像素坐标 ({x:.1f}, {y:.1f})，请在右侧输入框中填写该点信息")
            else:
                print("请先完成当前点的信息输入，或点击取消")
    
    def clear_temp_markers(self):
        """清除临时标记"""
        try:
            if hasattr(self, 'temp_point') and self.temp_point is not None:
                self.temp_point.remove()
                self.temp_point = None
        except (ValueError, AttributeError):
            pass
        
        try:
            if hasattr(self, 'temp_text') and self.temp_text is not None:
                self.temp_text.remove()
                self.temp_text = None
        except (ValueError, AttributeError):
            pass
    
    def confirm_add_point(self, event):
        """确认添加控制点"""
        if self.pending_point is None:
            print("请先在地图上点击一个位置")
            return
        
        try:
            # 获取输入的信息
            point_name = self.textbox_name.text.strip()
            lon_str = self.textbox_lon.text.strip()
            lat_str = self.textbox_lat.text.strip()
            
            if not point_name:
                point_name = f"控制点{len(self.control_points) + 1}"
            
            if not lon_str or not lat_str:
                print("请输入完整的经纬度坐标")
                return
            
            longitude = float(lon_str)
            latitude = float(lat_str)
            
            # 验证坐标合理性（中国境内）
            if not (70 <= longitude <= 140 and 15 <= latitude <= 55):
                print("警告：输入的坐标不在中国境内范围，请检查是否正确")
                return
            
            # 创建控制点
            control_point = {
                'pixel_x': self.pending_point['pixel_x'],
                'pixel_y': self.pending_point['pixel_y'],
                'lon': longitude,
                'lat': latitude,
                'name': point_name
            }
            
            self.control_points.append(control_point)
            
            # 清除临时标记
            self.clear_temp_markers()
            
            # 绘制正式的控制点
            self.draw_control_point(control_point['pixel_x'], control_point['pixel_y'], 
                                   len(self.control_points), point_name)
            
            print(f"成功添加控制点: {point_name} ({longitude:.6f}, {latitude:.6f})")
            
            # 重置状态
            self.pending_point = None
            self.set_input_enabled(False)
            
            # 更新状态显示
            self.update_calibration_status()
            
        except ValueError:
            print("经纬度格式错误，请输入有效的数字")
        except Exception as e:
            print(f"添加控制点时出错: {e}")
    
    def cancel_add_point(self, event):
        """取消添加控制点"""
        if self.pending_point is not None:
            # 清除临时标记
            self.clear_temp_markers()
            
            # 重置状态
            self.pending_point = None
            self.set_input_enabled(False)
            
            self.fig.canvas.draw()
            print("已取消添加控制点")
    
    def draw_control_point(self, x, y, point_id, name):
        """绘制控制点"""
        # 绘制红色圆点
        self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # 添加编号和名称标签
        self.ax.text(x+10, y-10, f"{point_id}\n{name}", fontsize=9, color='red',
                    weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig.canvas.draw()
    
    def clear_control_points(self, event):
        """清除所有控制点"""
        self.control_points = []
        self.pending_point = None
        self.set_input_enabled(False)
        self.redraw_calibration_view()
        print("已清除所有控制点")
    
    def remove_last_point(self, event):
        """删除最后一个控制点"""
        if self.control_points:
            removed = self.control_points.pop()
            self.redraw_calibration_view()
            print(f"已删除控制点: {removed['name']}")
        else:
            print("没有控制点可删除")
    
    def redraw_calibration_view(self):
        """重绘校准视图"""
        # 清除并重新显示图像
        self.ax.clear()
        self.ax.imshow(self.image_array)
        self.ax.set_title('地理坐标校准 - 请在已知位置点击并在右侧输入经纬度')
        
        # 重新绘制所有控制点
        for i, point in enumerate(self.control_points, 1):
            self.draw_control_point(point['pixel_x'], point['pixel_y'], i, point['name'])
        
        # 重新显示说明
        self.show_calibration_instructions()
        self.update_calibration_status()
        
        # 清除临时标记引用
        self.temp_point = None
        self.temp_text = None
    
    def finish_calibration(self, event):
        """完成校准"""
        if len(self.control_points) < 3:
            print("错误：至少需要3个控制点才能进行坐标转换")
            return
        
        print(f"\n校准完成！共使用 {len(self.control_points)} 个控制点")
        
        # 关闭校准界面
        plt.close(self.fig)
        
        # 计算转换参数
        self.calculate_transformation()
        
        return True
    
    def calculate_transformation(self):
        """计算坐标转换参数"""
        if len(self.control_points) < 3:
            print("控制点数量不足，无法计算转换参数")
            return False
        
        # 提取控制点数据
        pixel_coords = np.array([[p['pixel_x'], p['pixel_y']] for p in self.control_points])
        geo_coords = np.array([[p['lon'], p['lat']] for p in self.control_points])
        
        # 使用最小二乘法计算仿射变换参数
        # [lon, lat] = A * [x, y] + B
        
        # 添加齐次坐标
        pixel_homo = np.column_stack([pixel_coords, np.ones(len(pixel_coords))])
        
        # 计算变换矩阵 (对经度和纬度分别计算)
        self.transform_lon, _, _, _ = np.linalg.lstsq(pixel_homo, geo_coords[:, 0], rcond=None)
        self.transform_lat, _, _, _ = np.linalg.lstsq(pixel_homo, geo_coords[:, 1], rcond=None)
        
        print("坐标转换参数计算完成")
        
        # 计算转换精度
        self.calculate_transformation_accuracy()
        
        return True
    
    def calculate_transformation_accuracy(self):
        """计算转换精度"""
        total_error_lon = 0
        total_error_lat = 0
        
        print("\n转换精度验证:")
        print("控制点\t像素坐标\t\t真实坐标\t\t转换坐标\t\t误差(米)")
        print("-" * 80)
        
        for i, point in enumerate(self.control_points):
            # 使用转换参数计算该点的地理坐标
            pred_lon, pred_lat = self.pixel_to_geo(point['pixel_x'], point['pixel_y'])
            
            # 计算误差（大致转换为米）
            error_lon = abs(pred_lon - point['lon']) * 111000 * np.cos(np.radians(point['lat']))
            error_lat = abs(pred_lat - point['lat']) * 111000
            total_error = (error_lon**2 + error_lat**2)**0.5
            
            total_error_lon += error_lon
            total_error_lat += error_lat
            
            print(f"{point['name'][:8]:<8}\t({point['pixel_x']:.1f},{point['pixel_y']:.1f})\t\t"
                  f"({point['lon']:.6f},{point['lat']:.6f})\t"
                  f"({pred_lon:.6f},{pred_lat:.6f})\t{total_error:.1f}")
        
        avg_error_lon = total_error_lon / len(self.control_points)
        avg_error_lat = total_error_lat / len(self.control_points)
        avg_total_error = (avg_error_lon**2 + avg_error_lat**2)**0.5
        
        print(f"\n平均误差: {avg_total_error:.1f} 米")
        
        if avg_total_error > 1000:
            print("警告：转换误差较大，建议检查控制点坐标或增加控制点数量")
    
    def pixel_to_geo(self, pixel_x, pixel_y):
        """
        将像素坐标转换为经纬度坐标
        
        参数:
        pixel_x: 像素X坐标
        pixel_y: 像素Y坐标
        
        返回:
        (longitude, latitude): 经纬度坐标
        """
        if not hasattr(self, 'transform_lon') or not hasattr(self, 'transform_lat'):
            print("错误：尚未进行坐标校准")
            return None, None
        
        # 应用仿射变换
        pixel_homo = np.array([pixel_x, pixel_y, 1])
        longitude = np.dot(self.transform_lon, pixel_homo)
        latitude = np.dot(self.transform_lat, pixel_homo)
        
        return longitude, latitude
    
    def convert_csv_data(self, csv_file, output_file=None):
        """
        转换CSV文件中的像素坐标数据
        
        参数:
        csv_file: 输入CSV文件路径
        output_file: 输出文件路径（可选）
        
        返回:
        转换后的DataFrame
        """
        if not hasattr(self, 'transform_lon') or not hasattr(self, 'transform_lat'):
            print("错误：请先完成坐标校准")
            return None
        
        try:
            # 读取CSV数据
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"读取数据文件: {csv_file}")
            print(f"数据点数量: {len(df)}")
            
            # 检查必要的列
            required_columns = ['x', 'y']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 转换坐标
            longitudes = []
            latitudes = []
            
            print("正在转换坐标...")
            for index, row in df.iterrows():
                lon, lat = self.pixel_to_geo(row['x'], row['y'])
                longitudes.append(lon)
                latitudes.append(lat)
            
            # 添加经纬度列
            df['longitude'] = longitudes
            df['latitude'] = latitudes
            
            # 生成输出文件名
            if output_file is None:
                base_name = os.path.splitext(csv_file)[0]
                output_file = f"{base_name}_with_coords.csv"
            
            # 保存结果
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"转换完成，结果保存至: {output_file}")
            
            # 显示统计信息
            self.print_conversion_stats(df)
            
            return df
            
        except Exception as e:
            print(f"转换过程中出错: {e}")
            return None
    
    def print_conversion_stats(self, df):
        """打印转换统计信息"""
        print("\n转换统计:")
        print(f"总数据点: {len(df)}")
        
        if 'longitude' in df.columns and 'latitude' in df.columns:
            print(f"经度范围: {df['longitude'].min():.6f} ~ {df['longitude'].max():.6f}")
            print(f"纬度范围: {df['latitude'].min():.6f} ~ {df['latitude'].max():.6f}")
        
        if 'class_name' in df.columns:
            print("\n各类别分布:")
            class_counts = df['class_name'].value_counts()
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}个点")
    
    def save_calibration(self, filename="calibration.json"):
        """保存校准参数"""
        if not hasattr(self, 'transform_lon'):
            print("没有校准数据可保存")
            return
        
        import json
        
        calibration_data = {
            'image_path': self.image_path,
            'control_points': self.control_points,
            'transform_lon': self.transform_lon.tolist(),
            'transform_lat': self.transform_lat.tolist(),
            'image_size': [self.img_width, self.img_height]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, ensure_ascii=False, indent=2)
        
        print(f"校准参数已保存至: {filename}")
    
    def load_calibration(self, filename="calibration.json"):
        """加载校准参数"""
        try:
            import json
            
            with open(filename, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            self.control_points = calibration_data['control_points']
            self.transform_lon = np.array(calibration_data['transform_lon'])
            self.transform_lat = np.array(calibration_data['transform_lat'])
            
            print(f"校准参数已从 {filename} 加载")
            print(f"控制点数量: {len(self.control_points)}")
            return True
            
        except FileNotFoundError:
            print(f"校准文件 {filename} 不存在")
            return False
        except Exception as e:
            print(f"加载校准文件失败: {e}")
            return False

def find_image_files():
    """查找当前目录下的图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def find_csv_files():
    """查找当前目录下的CSV数据文件"""
    csv_files = []
    
    for file in os.listdir('.'):
        if file.lower().endswith('.csv') and file.startswith('data_'):
            csv_files.append(file)
    
    return csv_files

def main():
    """主函数"""
    print("像素坐标转经纬度转换工具")
    print("=" * 50)
    
    # 查找图片文件
    image_files = find_image_files()
    if not image_files:
        print("错误: 未找到图片文件")
        return
    
    print("可用图片文件:")
    for i, file in enumerate(image_files, 1):
        print(f"  {i}. {file}")
    
    # 选择图片
    try:
        choice = int(input(f"选择图片文件 (1-{len(image_files)}): ")) - 1
        if not (0 <= choice < len(image_files)):
            print("选择无效")
            return
        selected_image = image_files[choice]
    except (ValueError, KeyboardInterrupt):
        print("程序退出")
        return
    
    # 初始化转换器
    converter = CoordinateConverter(selected_image)
    
    # 检查是否有已保存的校准文件
    calibration_file = "calibration.json"
    if os.path.exists(calibration_file):
        use_existing = input(f"发现校准文件 {calibration_file}，是否使用？(y/N): ").lower()
        if use_existing == 'y':
            if converter.load_calibration(calibration_file):
                print("使用已保存的校准参数")
            else:
                print("加载校准文件失败，将重新校准")
                converter.setup_calibration_interface()
        else:
            print("将重新进行坐标校准")
            converter.setup_calibration_interface()
    else:
        print("开始坐标校准...")
        print("请在弹出的图像窗口中点击已知地理位置的点")
        converter.setup_calibration_interface()
    
    # 检查校准是否完成
    if not hasattr(converter, 'transform_lon'):
        print("校准未完成，程序退出")
        return
    
    # 保存校准参数
    save_calibration = input("是否保存校准参数供下次使用？(Y/n): ").lower()
    if save_calibration != 'n':
        converter.save_calibration(calibration_file)
    
    # 查找CSV数据文件
    csv_files = find_csv_files()
    if not csv_files:
        print("未找到数据文件")
        return
    
    print(f"\n可用数据文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    # 选择要转换的文件
    while True:
        try:
            choice_input = input(f"选择要转换的数据文件 (1-{len(csv_files)}, 或 'all' 转换所有文件, 'q' 退出): ").strip().lower()
            
            if choice_input == 'q':
                break
            elif choice_input == 'all':
                # 转换所有文件
                for csv_file in csv_files:
                    print(f"\n转换文件: {csv_file}")
                    converter.convert_csv_data(csv_file)
                break
            else:
                choice = int(choice_input) - 1
                if 0 <= choice < len(csv_files):
                    selected_csv = csv_files[choice]
                    print(f"\n转换文件: {selected_csv}")
                    converter.convert_csv_data(selected_csv)
                    
                    # 询问是否继续转换其他文件
                    continue_convert = input("是否转换其他文件？(y/N): ").lower()
                    if continue_convert != 'y':
                        break
                else:
                    print("选择无效，请重新输入")
                    
        except ValueError:
            print("输入格式错误，请重新输入")
        except KeyboardInterrupt:
            print("\n程序退出")
            break
    
    print("转换完成！")

if __name__ == "__main__":
    main()
