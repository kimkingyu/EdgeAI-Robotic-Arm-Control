# 香橙派 5 Pro 单舵机轻载安全测试指南 (免外接电源)

> 适合手头暂无独立电源时，直接借用香橙派 5V 引脚验证 I2C 通信与单舵机 0°~180° 平滑转动！

---

## 🔌 极简 5 根杜邦线接法

只接 **1 个舵机** 空载转动时，香橙派的 5V 引脚（最大可提供约 1A~1.5A 电流）足以驱动，接线如下：

| 香橙派 5 Pro 40-Pin 引脚 | 信号定义 | 连接到 ➔ | PCA9685 驱动板引脚 | 说明 |
| :---: | :---: | :---: | :---: | :---: |
| **Pin 3** | **I2C7_SDA** | ➔ | **SDA** | I2C 数据线 |
| **Pin 5** | **I2C7_SCL** | ➔ | **SCL** | I2C 时钟线 |
| **Pin 2** | **5V 电源** | ➔ | **VCC** | 芯片逻辑供电 (几毫安) |
| **Pin 4** | **5V 电源** | ➔ | **V+ (绿色端子正极)** | **单舵机动力供电 (跳线引出)** |
| **Pin 6** | **GND 地线** | ➔ | **GND** | 芯片与舵机共地 |

> 💡 **特别提醒**：
> 1. PCA9685 侧面排针上方有一个 **V+** 引脚，或者把杜邦线插进绿色大端子的 `+` 螺丝孔拧紧即可；
> 2. **千万只插 1 个舵机！** 舵机请插在 **通道 0 (Channel 0)**。切勿同时插 4~6 个舵机，否则电流过大香橙派会瞬间黑屏重启保护。

---

## ⚡ 首次测试命令 (0° -> 90° -> 180° -> 复位)

在板端终端直接敲这一行 Python 单行测试：

```bash
python3 -c "
import time
from src.controller.i2c_arm import I2CArmController

print('正在初始化 I2C 总线 7，连接 PCA9685...')
arm = I2CArmController(bus_num=7, address=0x40)
if arm.connect():
    print('测试通道 0 舵机转动: 0度 -> 90度 -> 180度')
    arm.pca.set_servo_angle(0, 0.0)
    time.sleep(1)
    arm.pca.set_servo_angle(0, 90.0)
    time.sleep(1)
    arm.pca.set_servo_angle(0, 180.0)
    time.sleep(1)
    arm.pca.set_servo_angle(0, 90.0)
    print('单舵机测试大获全胜！')
"
```
看到舵机平滑转动并回中，硬件通路就 100% 验证成功了！
