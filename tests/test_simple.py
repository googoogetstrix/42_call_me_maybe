import os
import pytest


# def test_simple(tmp_path):

#     out_dir = tmp_path / "data" / "out"
#     out_dir.mkdir(parents=True)

#     os.chmod(out_dir, 0o555)

#     try:
#         # 3. Execution & Verification: Expect an OSError or your custom error
#         with pytest.raises(PermissionError):
#             save_data(target_path=out_dir)
#     finally:
#         # Cleanup: Must restore permissions to allow pytest to delete the temp folder
#         os.chmod(out_dir, 0o777)        
#     assert 1 == 1
    
# def test_creates_directory_if_missing(tmp_path):
#     target = tmp_path / "new_folder"
    
#     # Call your function on a folder that doesn't exist yet
    
#     # Check if the folder was created automatically
#     assert target.exists()
#     assert target.is_dir()

def test_something():
    assert 1 == 1.0
