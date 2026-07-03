---
name: JNA HGDIOBJ missing from WinDef
description: WinDef.HGDIOBJ does not exist in JNA 5.14.0; must be defined locally as a PointerType subclass inside EdokitGdi32.
---

## Rule
`com.sun.jna.platform.win32.WinDef.HGDIOBJ` is absent from JNA 5.14.0 (and surrounding versions). Any GDI binding that references it will fail to compile.

**Why:** JNA platform's WinDef only exposes the most common typed handles (HWND, HDC, HBITMAP, HBRUSH, etc.). HGDIOBJ is the generic GDI supertype but was never added to the standard distribution.

**How to apply:** Define HGDIOBJ as a nested class inside the GDI JNA interface:
```java
class HGDIOBJ extends PointerType {
    public HGDIOBJ() {}
    public HGDIOBJ(Pointer p) { super(p); }
}
```
When passing a typed handle (e.g. HBITMAP) where HGDIOBJ is expected, wrap it:
`new HGDIOBJ(hBitmap.getPointer())`

This is done in `EdokitGdi32.java` (the fix is already applied).
