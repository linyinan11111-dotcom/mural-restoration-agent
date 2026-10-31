"""Build the small EXE launcher; the application uses the project runtime."""
from pathlib import Path
import subprocess
import sys
import shutil
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def main():
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    image = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 248, 248), radius=48, fill='#354f40')
    draw.rounded_rectangle((38, 38, 218, 218), radius=24, outline='#d9c598', width=5)
    for angle in range(0, 360, 45):
        import math
        x, y = 128 + 49*math.cos(math.radians(angle)), 128 + 49*math.sin(math.radians(angle))
        draw.ellipse((x-18, y-18, x+18, y+18), fill='#b9694b', outline='#e3cca0', width=3)
    draw.ellipse((102, 102, 154, 154), fill='#dfc68e')
    icon = assets / 'app.ico'
    image.save(icon, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--noconsole',
                    '--name', 'Shibi', '--icon', str(icon), '--distpath', str(ROOT / 'dist'),
                    '--workpath', str(ROOT / 'build'), '--specpath', str(ROOT / 'build'),
                    str(ROOT / 'desktop_launcher.py')], cwd=ROOT, check=True)
    shutil.copy2(ROOT / 'dist/Shibi.exe', ROOT / '拾壁.exe')
    print('Created: ' + str(ROOT / '拾壁.exe'))


if __name__ == '__main__':
    main()

