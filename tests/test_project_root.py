"""The project-root search must not stop at the filesystem root."""
import os

import pytest
from romda.utils import get_project_root


def test_finds_folder_in_ancestor(tmp_path):
    (tmp_path / 'src').mkdir()
    start = tmp_path / 'a' / 'b'
    start.mkdir(parents=True)
    assert get_project_root(str(start)) == str(tmp_path.resolve())


@pytest.mark.skipif(os.path.isdir('/src'), reason='needs no src folder at the filesystem root')
def test_filesystem_root_is_not_a_project(tmp_path):
    # searching '/' would return it for every directory outside a project
    # if the filesystem root held a src folder
    with pytest.raises(FileNotFoundError):
        get_project_root(str(tmp_path))
