# Team Member Profile Pictures

## How to Use

1. **Upload your team member photos to this folder**
   - Name them clearly (e.g., `student1.jpg`, `john_doe.png`)
   - Supported formats: JPG, PNG, GIF

2. **Update the team_members list in `home.py`**
   
   Find this section in `home.py`:
   ```python
   team_members = [
       {"name": "Student Name 1", "student_number": "12345678", "image": None},
       {"name": "Student Name 2", "student_number": "23456789", "image": None},
       {"name": "Student Name 3", "student_number": "34567890", "image": None},
       {"name": "Student Name 4", "student_number": "45678901", "image": None}
   ]
   ```

3. **Update with your information**
   ```python
   team_members = [
       {"name": "John Doe", "student_number": "20230001", "image": "john.jpg"},
       {"name": "Jane Smith", "student_number": "20230002", "image": "jane.jpg"},
       {"name": "Bob Johnson", "student_number": "20230003", "image": "bob.jpg"},
       {"name": "Alice Williams", "student_number": "20230004", "image": "alice.jpg"}
   ]
   ```

## Image Guidelines

- **Recommended size**: 200x200 pixels or larger (square format)
- **File size**: Keep under 1MB for faster loading
- **Format**: JPG or PNG preferred
- **Background**: Solid color or professional background works best
- **Face**: Clear, well-lit photo with face clearly visible

## Example File Structure

```
pics/
├── README.md          # This file
├── john.jpg           # Student 1's photo
├── jane.jpg           # Student 2's photo
├── bob.jpg            # Student 3's photo
└── alice.jpg          # Student 4's photo
```

## Notes

- If no image is provided (`image: None`), a placeholder icon (👤) will be displayed
- Images are automatically cropped to circular format
- The system will check if the file exists before displaying it
