"""Consistent local archives and safe restoration into a new directory."""
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
import json
from .storage import timestamp
from .imaging import sha256


def backup_workspace(store, destination):
    if any(j['status'] in {'queued', 'planning', 'running', 'evaluating'} for j in store.list('job')):
        raise ValueError('请等待任务结束后备份工作空间。')
    destination = Path(destination).resolve()
    if destination.is_relative_to(store.root.resolve()):
        raise ValueError('备份请保存到数据目录之外。')
    hashes = {}
    temporary = destination.with_suffix('.partial')
    with tempfile.TemporaryDirectory() as folder:
        snapshot = Path(folder)/'workbench.sqlite3'
        store.backup(snapshot)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output:
            output.write(snapshot, 'workbench.sqlite3')
            hashes['workbench.sqlite3'] = sha256(snapshot)
            for directory in store.root.iterdir():
                if not directory.is_dir() or len(directory.name) != 32 or any(c not in '0123456789abcdef' for c in directory.name):
                    continue
                for path in directory.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        relative = path.relative_to(store.root).as_posix()
                        output.write(path, relative)
                        hashes[relative] = sha256(path)
            output.writestr('backup.json', json.dumps({'format': 1, 'created_at': timestamp(), 'sha256': hashes}, indent=2))
    temporary.replace(destination)
    return str(destination)


def restore_workspace(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('恢复目标必须是尚不存在的新目录。')
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('backup.json'))
        if manifest.get('format') != 1 or sum(i.file_size for i in source.infolist()) > 100*1024**3:
            raise ValueError('不支持的备份格式或超过 100GB。')
        for name in source.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('备份包含不安全路径。')
            if name not in manifest['sha256'] and name != 'backup.json':
                raise ValueError('备份含未登记文件。')
        destination.mkdir(parents=True)
        for name, expected in manifest['sha256'].items():
            # Validate manifest paths separately, including names missing from ZIP listing.
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('清单路径无效。')
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))
            if sha256(target) != expected:
                raise ValueError('备份文件校验失败：'+name)
    return destination
