# tts模型
https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/tts.html

# asr模型


# unbutu20.04 apt安装python3.10 apt没有相关仓库支持3.10
```shell
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libsqlite3-dev libreadline-dev libffi-dev curl
sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev libncurses5-dev libgdbm-dev \
libnss3-dev libffi-dev liblzma-dev

# 下载 Python 3.10 源码，版本号可自行选择
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xf Python-3.10.0.tgz
cd Python-3.10.0

./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall




update-alternatives 是 Debian/Ubuntu 系统原生的替代方案管理系统，可安全注册多个 Python 解释器路径，并通过交互式命令统一调度，默认不修改任何系统关键文件。

1、确认当前已安装的 Python 3 可执行路径：
ls -l /usr/bin/python3*

2、将 Python 3.8 注册为备选版本（优先级设为 1）：
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

3、将 Python 3.10 注册为备选版本（优先级设为 2 越大越优先）：
 sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 2
4、启动交互式版本选择：
sudo update-alternatives --config python3



wget https://bootstrap.pypa.io/get-pip.py

# 用你的 Python 3.10 执行
/home/tyt/Python-3.10.0/bin/python3.10 get-pip.py

# 删除脚本（可选）
rm get-pip.py

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

sudo update-alternatives --install /usr/bin/pip3 pip3 /home/tyt/.local/bin/pip3.10 100
sudo update-alternatives --config pip3

```

## 将MeloTTS标记为源码目录

```shell
迅雷下载 https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip
3. 修改download.py文件
• 定位文件：nano /usr/local/lib/python3.9/site-packages/unidic/download.py
• 修改点1：注释第93行download_version函数中除download_and_clean()外的代码：def download_version(version, dest_dir): # 注释以下内容（示例）： # if not dest_dir: # dest_dir = get_default_dicdir() # ...其他代码... download_and_clean('3.1.0', '') # 保留此行并设置参数
• 修改点2：注释第63行的download_process调用：# download_process(url, dest_path) # 注释此行
4. 执行下载命令

```