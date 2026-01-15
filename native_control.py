"""Native control module for MiniOrca.

Provides:
- Mouse control (move, click, drag, scroll)
- Keyboard control (type, hotkeys, key press/release)
- Screen reading (screenshots, OCR, element detection)
- Application control (window management)
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
import os
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

# Module logger
logger = logging.getLogger(__name__)


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class SpecialKey(Enum):
    """Special keyboard keys."""
    ENTER = "enter"
    TAB = "tab"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    DELETE = "delete"
    SPACE = "space"
    
    # Arrow keys
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    
    # Modifiers
    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    SUPER = "super"  # Windows/Command key
    
    # Function keys
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    F9 = "f9"
    F10 = "f10"
    F11 = "f11"
    F12 = "f12"
    
    # Other
    HOME = "home"
    END = "end"
    PAGEUP = "pageup"
    PAGEDOWN = "pagedown"
    INSERT = "insert"
    PRINTSCREEN = "printscreen"
    CAPSLOCK = "capslock"
    NUMLOCK = "numlock"


@dataclass
class ScreenRegion:
    """A region on the screen."""
    x: int
    y: int
    width: int
    height: int
    
    def contains(self, px: int, py: int) -> bool:
        return (self.x <= px < self.x + self.width and 
                self.y <= py < self.y + self.height)
    
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class WindowInfo:
    """Information about a window."""
    id: str
    title: str
    app_name: str
    region: ScreenRegion
    is_active: bool


class MouseController:
    """Control mouse movements and clicks."""
    
    def __init__(self):
        self._pyautogui = None
        self._load_backend()
    
    def _load_backend(self):
        """Load mouse control backend."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move to corner to abort
            pyautogui.PAUSE = 0.05     # Small pause between actions
            self._pyautogui = pyautogui
        except (ImportError, Exception) as e:
            # Handle missing DISPLAY or other GUI-related errors in headless environments
            logger.warning(f"Failed to load pyautogui backend: {e}")
            self._pyautogui = None
    
    async def move(self, x: int, y: int, duration: float = 0.25) -> None:
        """Move mouse to absolute position."""
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.moveTo, x, y, duration=duration)
        else:
            await self._xdotool_move(x, y)
    
    async def move_relative(self, dx: int, dy: int, duration: float = 0.25) -> None:
        """Move mouse relative to current position."""
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.move, dx, dy, duration=duration)
        else:
            current = await self.get_position()
            await self._xdotool_move(current[0] + dx, current[1] + dy)
    
    async def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1
    ) -> None:
        """Click at position (or current position if not specified)."""
        if self._pyautogui:
            await asyncio.to_thread(
                self._pyautogui.click,
                x=x, y=y,
                button=button.value,
                clicks=clicks
            )
        else:
            if x is not None and y is not None:
                await self._xdotool_move(x, y)
            await self._xdotool_click(button, clicks)
    
    async def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Double-click at position."""
        await self.click(x, y, clicks=2)
    
    async def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Right-click at position."""
        await self.click(x, y, button=MouseButton.RIGHT)
    
    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5
    ) -> None:
        """Drag from start to end position."""
        if self._pyautogui:
            await asyncio.to_thread(
                self._pyautogui.moveTo, start_x, start_y
            )
            await asyncio.to_thread(
                self._pyautogui.drag,
                end_x - start_x, end_y - start_y,
                duration=duration,
                button=button.value
            )
        else:
            # xdotool drag
            await self._xdotool_move(start_x, start_y)
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "mousedown", "1"],
                capture_output=True
            )
            await self._xdotool_move(end_x, end_y)
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "mouseup", "1"],
                capture_output=True
            )
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll up (positive) or down (negative)."""
        if x is not None and y is not None:
            await self.move(x, y)
        
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.scroll, clicks)
        else:
            direction = "4" if clicks > 0 else "5"
            for _ in range(abs(clicks)):
                await asyncio.to_thread(
                    subprocess.run,
                    ["xdotool", "click", direction],
                    capture_output=True
                )
    
    async def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        if self._pyautogui:
            pos = self._pyautogui.position()
            return (pos.x, pos.y)
        else:
            result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getmouselocation"],
                capture_output=True,
                text=True
            )
            # Parse "x:123 y:456 screen:0 window:12345"
            parts = result.stdout.strip().split()
            x = int(parts[0].split(":")[1])
            y = int(parts[1].split(":")[1])
            return (x, y)
    
    async def _xdotool_move(self, x: int, y: int) -> None:
        """Move using xdotool."""
        await asyncio.to_thread(
            subprocess.run,
            ["xdotool", "mousemove", str(x), str(y)],
            capture_output=True
        )
    
    async def _xdotool_click(self, button: MouseButton, clicks: int) -> None:
        """Click using xdotool."""
        btn_map = {
            MouseButton.LEFT: "1",
            MouseButton.MIDDLE: "2",
            MouseButton.RIGHT: "3",
        }
        for _ in range(clicks):
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "click", btn_map[button]],
                capture_output=True
            )


class KeyboardController:
    """Control keyboard input."""
    
    def __init__(self):
        self._pyautogui = None
        self._load_backend()
    
    def _load_backend(self):
        try:
            import pyautogui
            self._pyautogui = pyautogui
        except (ImportError, Exception) as e:
            # Handle missing DISPLAY or other GUI-related errors in headless environments
            logger.warning(f"Failed to load pyautogui backend: {e}")
            self._pyautogui = None
    
    async def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type text as if from keyboard."""
        if self._pyautogui:
            await asyncio.to_thread(
                self._pyautogui.write, text, interval=interval
            )
        else:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "type", "--clearmodifiers", "--delay", str(int(interval * 1000)), text],
                capture_output=True
            )
    
    async def press(self, key: str | SpecialKey) -> None:
        """Press and release a key."""
        key_str = key.value if isinstance(key, SpecialKey) else key
        
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.press, key_str)
        else:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "key", key_str],
                capture_output=True
            )
    
    async def key_down(self, key: str | SpecialKey) -> None:
        """Press a key down (without releasing)."""
        key_str = key.value if isinstance(key, SpecialKey) else key
        
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.keyDown, key_str)
        else:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "keydown", key_str],
                capture_output=True
            )
    
    async def key_up(self, key: str | SpecialKey) -> None:
        """Release a key."""
        key_str = key.value if isinstance(key, SpecialKey) else key
        
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.keyUp, key_str)
        else:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "keyup", key_str],
                capture_output=True
            )
    
    async def hotkey(self, *keys: str | SpecialKey) -> None:
        """Press a hotkey combination (e.g., Ctrl+C)."""
        key_strs = [k.value if isinstance(k, SpecialKey) else k for k in keys]
        
        if self._pyautogui:
            await asyncio.to_thread(self._pyautogui.hotkey, *key_strs)
        else:
            combo = "+".join(key_strs)
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "key", combo],
                capture_output=True
            )


class ScreenReader:
    """Read and analyze screen content."""
    
    def __init__(self):
        self._pyautogui = None
        try:
            import pyautogui
            self._pyautogui = pyautogui
        except (ImportError, Exception) as e:
            # Handle missing DISPLAY or other GUI-related errors in headless environments
            logger.warning(f"Failed to load pyautogui backend: {e}")
            self._pyautogui = None
    
    async def screenshot(
        self,
        region: Optional[ScreenRegion] = None
    ) -> bytes:
        """Take a screenshot, returns PNG bytes."""
        if self._pyautogui:
            import io
            
            if region:
                img = await asyncio.to_thread(
                    self._pyautogui.screenshot,
                    region=(region.x, region.y, region.width, region.height)
                )
            else:
                img = await asyncio.to_thread(self._pyautogui.screenshot)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            # Use scrot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
            
            try:
                if region:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["scrot", "-a", f"{region.x},{region.y},{region.width},{region.height}", temp_path],
                        capture_output=True
                    )
                else:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["scrot", temp_path],
                        capture_output=True
                    )
                
                with open(temp_path, 'rb') as f:
                    return f.read()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    async def screenshot_base64(self, region: Optional[ScreenRegion] = None) -> str:
        """Take a screenshot, returns base64 data URL."""
        png_bytes = await self.screenshot(region)
        b64 = base64.b64encode(png_bytes).decode()
        return f"data:image/png;base64,{b64}"
    
    async def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        if self._pyautogui:
            size = self._pyautogui.size()
            return (size.width, size.height)
        else:
            result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getdisplaygeometry"],
                capture_output=True,
                text=True
            )
            parts = result.stdout.strip().split()
            return (int(parts[0]), int(parts[1]))
    
    async def locate_on_screen(self, image_path: str, confidence: float = 0.9) -> Optional[ScreenRegion]:
        """Find an image on screen."""
        if self._pyautogui:
            try:
                loc = await asyncio.to_thread(
                    self._pyautogui.locateOnScreen,
                    image_path,
                    confidence=confidence
                )
                if loc:
                    return ScreenRegion(loc.left, loc.top, loc.width, loc.height)
            except Exception:
                pass
        return None
    
    async def ocr(self, region: Optional[ScreenRegion] = None) -> str:
        """Perform OCR on screen or region."""
        # Take screenshot
        png_bytes = await self.screenshot(region)
        
        # Try tesseract
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(png_bytes)
                temp_path = f.name
            
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["tesseract", temp_path, "stdout"],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
            finally:
                os.unlink(temp_path)
        except FileNotFoundError:
            return "[OCR not available - install tesseract]"


class WindowController:
    """Control and manage windows."""
    
    async def get_active_window(self) -> Optional[WindowInfo]:
        """Get information about the active window."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True
            )
            window_id = result.stdout.strip()
            
            if window_id:
                return await self.get_window_info(window_id)
        except Exception:
            pass
        return None
    
    async def get_window_info(self, window_id: str) -> Optional[WindowInfo]:
        """Get information about a specific window."""
        try:
            # Get window name
            name_result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getwindowname", window_id],
                capture_output=True,
                text=True
            )
            title = name_result.stdout.strip()
            
            # Get window geometry
            geo_result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getwindowgeometry", window_id],
                capture_output=True,
                text=True
            )
            
            # Parse geometry
            lines = geo_result.stdout.strip().split('\n')
            pos_line = [l for l in lines if 'Position:' in l][0]
            size_line = [l for l in lines if 'Geometry:' in l][0]
            
            # "  Position: 100,200 (screen: 0)"
            pos_parts = pos_line.split(':')[1].strip().split()[0].split(',')
            x, y = int(pos_parts[0]), int(pos_parts[1])
            
            # "  Geometry: 800x600"
            size_parts = size_line.split(':')[1].strip().split('x')
            width, height = int(size_parts[0]), int(size_parts[1])
            
            # Get active window to check if this is active
            active = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True
            )
            is_active = active.stdout.strip() == window_id
            
            return WindowInfo(
                id=window_id,
                title=title,
                app_name=title.split(' - ')[-1] if ' - ' in title else title,
                region=ScreenRegion(x, y, width, height),
                is_active=is_active,
            )
        except Exception:
            return None
    
    async def list_windows(self) -> List[WindowInfo]:
        """List all windows."""
        windows = []
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "search", "--onlyvisible", "--name", ""],
                capture_output=True,
                text=True
            )
            
            for window_id in result.stdout.strip().split('\n'):
                if window_id:
                    info = await self.get_window_info(window_id)
                    if info:
                        windows.append(info)
        except Exception:
            pass
        
        return windows
    
    async def activate_window(self, window_id: str) -> bool:
        """Bring a window to focus."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "windowactivate", window_id],
                capture_output=True
            )
            return True
        except Exception:
            return False
    
    async def move_window(self, window_id: str, x: int, y: int) -> bool:
        """Move a window to position."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "windowmove", window_id, str(x), str(y)],
                capture_output=True
            )
            return True
        except Exception:
            return False
    
    async def resize_window(self, window_id: str, width: int, height: int) -> bool:
        """Resize a window."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "windowsize", window_id, str(width), str(height)],
                capture_output=True
            )
            return True
        except Exception:
            return False
    
    async def close_window(self, window_id: str) -> bool:
        """Close a window."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "windowclose", window_id],
                capture_output=True
            )
            return True
        except Exception:
            return False
    
    async def search_window(self, name: str) -> Optional[str]:
        """Search for a window by name."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "search", "--name", name],
                capture_output=True,
                text=True
            )
            windows = result.stdout.strip().split('\n')
            return windows[0] if windows and windows[0] else None
        except Exception:
            return None


class NativeController:
    """Unified native control interface for MiniOrca."""
    
    def __init__(self):
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.screen = ScreenReader()
        self.window = WindowController()
    
    async def describe_screen(self) -> Dict[str, Any]:
        """Get a description of the current screen state."""
        screen_size = await self.screen.get_screen_size()
        mouse_pos = await self.mouse.get_position()
        active_window = await self.window.get_active_window()
        
        return {
            "screen_size": {"width": screen_size[0], "height": screen_size[1]},
            "mouse_position": {"x": mouse_pos[0], "y": mouse_pos[1]},
            "active_window": active_window.__dict__ if active_window else None,
        }
    
    async def execute_action(self, action: Dict[str, Any], caller_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a native action from a structured command.

        This method now emits audit logs for request and execution. `caller_meta` may include caller IP, user id, or headers.

        Actions:
        - mouse_move: {"x": int, "y": int}
        - mouse_click: {"x": int?, "y": int?, "button": str?, "clicks": int?}
        - mouse_drag: {"start_x": int, "start_y": int, "end_x": int, "end_y": int}
        - mouse_scroll: {"clicks": int, "x": int?, "y": int?}
        - keyboard_type: {"text": str}
        - keyboard_press: {"key": str}
        - keyboard_hotkey: {"keys": [str, ...]}
        - screenshot: {"region": {"x": int, "y": int, "width": int, "height": int}?}
        - window_activate: {"title": str}
        - window_close: {"title": str}
        """
        action_id = str(uuid.uuid4())
        start_ts = time.time()
        action_type = action.get("type")
        result = {"success": False, "action": action_type, "action_id": action_id}

        # Audit: requested
        try:
            logger.info("", extra={"payload": {
                "event": "native_action_requested",
                "action_id": action_id,
                "timestamp": start_ts,
                "action": action,
                "caller_meta": caller_meta,
            }})
        except Exception:
            logger.exception("Failed to emit native_action_requested log")

        try:
            if action_type == "mouse_move":
                await self.mouse.move(action["x"], action["y"])
                result["success"] = True

            elif action_type == "mouse_click":
                button = MouseButton(action.get("button", "left"))
                await self.mouse.click(
                    action.get("x"),
                    action.get("y"),
                    button,
                    action.get("clicks", 1)
                )
                result["success"] = True

            elif action_type == "mouse_drag":
                await self.mouse.drag(
                    action["start_x"],
                    action["start_y"],
                    action["end_x"],
                    action["end_y"]
                )
                result["success"] = True

            elif action_type == "mouse_scroll":
                await self.mouse.scroll(
                    action["clicks"],
                    action.get("x"),
                    action.get("y")
                )
                result["success"] = True

            elif action_type == "keyboard_type":
                await self.keyboard.type_text(action["text"])
                result["success"] = True

            elif action_type == "keyboard_press":
                await self.keyboard.press(action["key"])
                result["success"] = True

            elif action_type == "keyboard_hotkey":
                await self.keyboard.hotkey(*action["keys"])
                result["success"] = True

            elif action_type == "screenshot":
                region = None
                if "region" in action:
                    r = action["region"]
                    region = ScreenRegion(r["x"], r["y"], r["width"], r["height"])
                img = await self.screen.screenshot_base64(region)
                result["success"] = True
                result["image"] = img

            elif action_type == "window_activate":
                window_id = await self.window.search_window(action["title"])
                if window_id:
                    result["success"] = await self.window.activate_window(window_id)

            elif action_type == "window_close":
                window_id = await self.window.search_window(action["title"])
                if window_id:
                    result["success"] = await self.window.close_window(window_id)

            else:
                result["error"] = f"Unknown action type: {action_type}"

        except Exception as e:
            result["error"] = str(e)

        # Audit: executed
        try:
            end_ts = time.time()
            logger.info("", extra={"payload": {
                "event": "native_action_executed",
                "action_id": action_id,
                "action": action,
                "result": result,
                "duration_seconds": end_ts - start_ts,
                "caller_meta": caller_meta,
            }})
        except Exception:
            logger.exception("Failed to emit native_action_executed log")

        return result


# Global controller instance
_native_controller: Optional[NativeController] = None


def get_native_controller(allow_create: bool = False) -> Optional[NativeController]:
    """Get the global native controller instance.

    If environment variable `ENABLE_NATIVE_CONTROL` is not '1', this returns None
    unless `allow_create` is True (useful for tests/explicit creation).
    """
    global _native_controller
    enabled = os.getenv("ENABLE_NATIVE_CONTROL", "0") == "1"
    if not enabled and not allow_create:
        return None
    if _native_controller is None:
        _native_controller = NativeController()
    return _native_controller
