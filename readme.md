# 架构设计

## 记忆
### 生【人】物记忆
### 时间记忆
### 场景空间记忆
### 大事记

## 情绪
### 情绪起止时间，延续时间，情绪转换时间点
### 情绪分类 ekman plutchik 
### 基础情绪 Google goemotions 
### 情绪指数 【0-1】不强烈到强烈

## 通用基础知识与技能
### 采用云端mcp接入

## 唤醒方案[预制关键词，自动唤醒]

## 陪伴对象
### 姓名 出生日期，性别，升高，体重，学历，音色，与主陪伴人关系

## 性格
### todo

## 预制模型
### 硬件mcu[机器人底座] mpu【暂定rk35xx系列】
### tts模型 qwen3-tts
### asr模型 vad检测 sherpa-onnx senseVoice 
### online llm模型 api 









# 模型处理
ubuntu22.04安装 pyaudio 【配置安装镜像】
sudo apt install python3-pyaudio

镜像配置
```shell
sudo nano /etc/apt/sources.list

# 阿里云镜像源 - Ubuntu 22.04 (Jammy)
deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-proposed main restricted universe multiverse

sudo apt update
# 直接pip install 底层缺失操作系统相关c的支持
sudo apt install python3-pyaudio

sudo apt install ffmpeg 
```


音频设备桥接

sudo apt install pulseaudio


export PULSE_SERVER=127.0.0.1
export AUDIODEV=default

创建或编辑 ~/.asoundrc）：

```text
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
```

python环境选择


操作系统升级
wsl --shutdown
# 查看你的实例名称，通常为 Ubuntu-22.04
wsl -l -v

# 执行导出
PS C:\Windows\System32> wsl -l -v
  NAME            STATE           VERSION
* Ubuntu-20.04    Running         2 
* 
wsl --export Ubuntu-22.04 "D:\wsl-backups\Ubuntu-22.04-backup.tar"

sudo apt update && sudo apt full-upgrade -y

确认升级配置：编辑配置文件，确保系统被设置为仅升级到 LTS 版本。

bash
sudo nano /etc/update-manager/release-upgrades
找到 Prompt=lts 这一行，如果被注释或值为 never，请修改为 Prompt=lts，然后保存退出。

执行版本升级：运行升级命令。过程中会提示确认，按 y 继续即可。

bash
sudo do-release-upgrade
静默升级
sudo do-release-upgrade -f DistUpgradeViewNonInteractive
注意：升级过程可能需要较长时间，请耐心等待。

重启并验证：升级完成后，需要重启 WSL 实例，然后检查版本号。

bash
# 退出当前会话，然后在 PowerShell 中重启
wsl --terminate Ubuntu-22.04
wsl -d Ubuntu-22.04



venv环境激活
 cd /home/tyt/.virtualenvs/py312/bin
source  ./activate
