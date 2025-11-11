"""
Демонстрация алгоритма Z-буфера (software rasterizer на NumPy + Tkinter)

Краткий обзор конвейера рендеринга (high-level):
- Геометрия сцены — набор многогранников (Polyhedron) из main.py.
- Камера — класс Camera из main.py, предоставляет матрицу вида (view) и проекции (projection).
- Преобразования: вершины переводятся в координаты камеры (view), затем в clip-space (projection),
  затем выполняется перспективное деление (perspective divide) и маппинг в экранные координаты.
- Z-буфер: для каждого пикселя хранится наименьшая глубина (минимальная z в системе камеры),
  используем глубину из КАМЕРНОГО пространства до перспективного деления (важно: не из clip-space).

Формальный алгоритм Z-буфера (соответствие по коду):
1) clear_buffers: обнуляем буфер кадра (цветов) и заполняем Z-буфер +∞.
2) для каждой грани (в нашем случае — многоугольник разбивается на треугольники «веером»):
3) draw_face → draw_triangle: растеризация в пределах bounding-box пикселей.
4) draw_triangle: для каждого пикселя внутри треугольника вычисляем глубину z(x,y) с
   перспективно-корректной интерполяцией: интерполируются 1/z вершин, затем z = 1 / (u/z0 + v/z1 + w/z2).
5) draw_triangle: сравниваем z(x,y) с текущим значением Z-буфера.
6) Если z меньше — обновляем цвет и Z-буфер.
7) Иначе — пиксель оставляем без изменений.

Подсистемы и нюансы:
- Отсечение нелицевых граней (backface culling) делегировано общей функции из main.py
  compute_visible_faces_camera_space, чтобы поведение совпадало между приложениями.
- Перспективная проекция: используется Camera.get_projection_matrix (фокусное из FOV),
  параллельная проекция — ortho_xy() с адаптивным масштабом для стабильности при «отдалении» камеры.
- Быстрые проверки: отбрасываем треугольники с вершинами с depth <= 0 (за камерой),
  и bounding-box полностью вне экрана.

Ограничения (намеренно упрощено в учебных целях):
- Не реализован near-plane clipping (частично пересекающие ближнюю плоскость треугольники отбрасываются целиком),
  вместо аккуратного клиппинга по near-plane.
- Нет антиалиасинга и субпиксельной точности; raster step = 1 px.
- Нет аппаратного ускорения; для серьёзных сцен можно рассмотреть векторизацию и Numba.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
from main import (
    Polyhedron, hexahedron, tetrahedron, octahedron, 
    icosahedron, dodecahedron, Camera, T, S, Rx, Ry, Rz,
    ortho_xy, perspective, compute_visible_faces_camera_space
)


def clear_buffers(width, height):
    """Создаёт новые цветовой и Z-буферы заданного размера.

    Соответствие формальному алгоритму:
    1) заполнить буфер кадра фоновым значением (чёрный цвет)
    2) заполнить Z-буфер максимальным значением z (бесконечность)

    Форматы:
    - color_buffer: (H, W, 3) uint8, RGB
    - z_buffer: (H, W) float32, инициализируется +inf, сравнение идёт на «меньше»
    """
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    return color_buffer, z_buffer



def barycentric(p, a, b, c):
    """Возвращает барицентрические координаты (u, v, w) точки p в треугольнике (a,b,c).

    Свойства:
    - u + v + w = 1; p = u*a + v*b + w*c;
    - если какая-либо из (u,v,w) < 0 — p вне треугольника (для строгой проверки);
    - при вырожденном треугольнике возвращает (-1, -1, -1).
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return -1, -1, -1

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def draw_triangle(p0, p1, p2, z0, z1, z2, color, color_buffer, z_buffer, enable_zbuffer=True):
    """Растеризует треугольник с Z-буфером.

    Шаги 3–7 формального алгоритма выполняются здесь:
    3) перевод многоугольника в растровую форму (обход пикселей в bounding box)
    4) для каждого пикселя внутри треугольника вычислить глубину z(x,y)
    5) сравнить z(x,y) с Z-буфером[x,y]
    6) если z меньше — записать цвет и обновить Z-буфер
    7) иначе — ничего не делать

    Перспективно-корректная интерполяция глубины:
    - интерполируем величины invz_i = 1/z_i с барицентрическими весами u,v,w;
      затем z = 1 / (u*invz0 + v*invz1 + w*invz2).
    - Это соответствует линейности по экрану атрибутов в пространстве 1/z и
      устраняет искажения при перспективе.

    Оптимизации/проверки:
    - Ранний отсев треугольников, если любая вершина имеет depth<=0 (за камерой).
    - Быстрая проверка на выход за экран через bounding box.
    """
    height, width = z_buffer.shape
    # Отбрасываем треугольники, у которых вершины за камерой (depth <= 0)
    eps = 1e-6
    if (z0 <= eps) or (z1 <= eps) or (z2 <= eps):
        return

    # Bounding box
    bb_min_x = min(p0[0], p1[0], p2[0])
    bb_max_x = max(p0[0], p1[0], p2[0])
    bb_min_y = min(p0[1], p1[1], p2[1])
    bb_max_y = max(p0[1], p1[1], p2[1])
    if bb_max_x < 0 or bb_min_x >= width or bb_max_y < 0 or bb_min_y >= height:
        return
    min_x = max(0, int(bb_min_x))
    max_x = min(width - 1, int(bb_max_x))
    min_y = max(0, int(bb_min_y))
    max_y = min(height - 1, int(bb_max_y))

    a = np.array([p0[0], p0[1]], dtype=np.float32)
    b = np.array([p1[0], p1[1]], dtype=np.float32)
    c = np.array([p2[0], p2[1]], dtype=np.float32)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x, y], dtype=np.float32)
            u, v, w = barycentric(p, a, b, c)
            if u >= -1e-5 and v >= -1e-5 and w >= -1e-5:
                # Шаг 4: вычисление глубины по пикселю через перспективно-корректную интерполяцию
                invz0 = 1.0 / z0
                invz1 = 1.0 / z1
                invz2 = 1.0 / z2
                invz = u * invz0 + v * invz1 + w * invz2
                if invz <= 0:
                    continue
                z = 1.0 / invz

                # Шаг 5–7: сравнение с Z-буфером и возможное обновление
                if enable_zbuffer:
                    if z < z_buffer[y, x]:
                        z_buffer[y, x] = z
                        color_buffer[y, x] = color
                else:
                    color_buffer[y, x] = color

def is_face_front_facing(face, vertices_camera):
    """Проверяет, лицевая ли грань в координатах камеры.

    Примечание: в актуальном рендере используется общая функция
    main.compute_visible_faces_camera_space. Этот хелпер оставлен как
    иллюстрация альтернативного критерия (эквивалентный по смыслу).

    В координатах камеры направление взгляда фиксировано вдоль -Z, поэтому грань
    считается лицевой, если dot(n_cam, view_dir) < 0, где view_dir = (0,0,-1).
    """
    idx = face.indices
    if len(idx) < 3:
        return False
    v0 = vertices_camera[:3, idx[0]]
    v1 = vertices_camera[:3, idx[1]]
    v2 = vertices_camera[:3, idx[2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    if np.linalg.norm(n) < 1e-10:
        return False
    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.dot(n, view_dir) < 0.0

def draw_face(face, vertices_screen, depths, color, color_buffer, z_buffer, enable_zbuffer=True):
    indices = face.indices
    if len(indices) < 3:
        return
    # Шаг 3: триангуляция многоугольника (веер) как часть перевода в растровую форму
    for i in range(1, len(indices) - 1):
        i0 = indices[0]
        i1 = indices[i]
        i2 = indices[i + 1]
        p0 = (vertices_screen[0, i0], vertices_screen[1, i0])
        p1 = (vertices_screen[0, i1], vertices_screen[1, i1])
        p2 = (vertices_screen[0, i2], vertices_screen[1, i2])
        z0 = depths[i0]
        z1 = depths[i1]
        z2 = depths[i2]
        draw_triangle(p0, p1, p2, z0, z1, z2, color, color_buffer, z_buffer, enable_zbuffer)

def render_polyhedron(polyhedron, view_matrix, proj_matrix, color, color_buffer, z_buffer,
                                            use_perspective=True, cull_backfaces=False, enable_zbuffer=True,
                                            camera: Camera | None = None):
        """Рендерит многогранник в заданные буферы (цвет и глубина).

        Контракт:
        - Вход: матрицы view и projection (раздельно), цвет RGB, флаги use_perspective/cull/zbuffer.
        - Глубина для Z-теста: depth = -z_cam, где z_cam — координата вершины в пространстве камеры.
        - Проекция влияет только на экранные координаты; глубина берётся ДО перспективного деления.

        Отсечение нелицевых граней:
        - Используем общую функцию main.compute_visible_faces_camera_space (единый источник истины),
            чтобы критерий совпадал с приложением main.py.
        - Критерий: dot(normal, view_to_face) < 0, где view_to_face = normalize(центр грани) в перспективе
            и (0,0,1) — в орто-проекции (в системе камеры).

        Ограничения:
        - Near-plane clipping не выполнен: треугольник с вершиной за камерой отбрасывается целиком.
        - Для сложных многоугольников треугольникуем «веером» (fan triangulation).
        """
        # 1) Переход в пространство камеры и вычисление видимых граней — через main.compute_visible_faces_camera_space
        if cull_backfaces and camera is not None:
                P_cam, visible_faces = compute_visible_faces_camera_space(polyhedron, camera, ortho=(not use_perspective))
        else:
                P_cam = polyhedron.copy().apply(view_matrix)
                visible_faces = P_cam.faces

        vertices_camera = P_cam.V  # 4xN в камере
        camera_z = vertices_camera[2, :]
        depths = -camera_z

        # 3) Проекция -> экранные координаты
        # Перспективное деление и получение (x,y) выполняем через общий метод Polyhedron.projected из main.py
        height, width = z_buffer.shape
        x_ndc, y_ndc = P_cam.projected(proj_matrix)
        scale = min(width, height) * 0.3
        screen_x = (x_ndc * scale + width / 2).astype(np.int32)
        screen_y = (-y_ndc * scale + height / 2).astype(np.int32)
        vertices_screen = np.vstack([screen_x, screen_y])

        # 4) Растеризация только видимых граней
        for face in visible_faces:
                draw_face(face, vertices_screen, depths, color, color_buffer, z_buffer, enable_zbuffer=enable_zbuffer)


class Scene:
    """Класс для представления сцены с несколькими объектами"""
    
    def __init__(self, name, objects, camera_pos, camera_target):
        """
        name: название сцены
        objects: список кортежей (polyhedron, color)
        camera_pos: начальная позиция камеры
        camera_target: цель камеры
        """
        self.name = name
        self.objects = objects
        self.camera_pos = camera_pos
        self.camera_target = camera_target





class ZBufferApp:
    """Главное приложение для демонстрации Z-буфера"""
    
    def __init__(self, root):
        self.root = root
        self.root.title('Демонстрация алгоритма Z-буфера')
        
        # Параметры
        self.use_zbuffer = tk.BooleanVar(value=True)
        self.projection_mode = tk.StringVar(value='perspective') 
        self.current_scene_idx = 0
        self.use_culling = tk.BooleanVar(value=True)
        
        # Создаем сцены
        self.scenes = self.create_scenes()
        
        # Камера
        scene = self.scenes[self.current_scene_idx]
        self.camera = Camera(
            position=scene.camera_pos,
            target=scene.camera_target,
            fov=60.0
        )
        
        # Рендерер (будет инициализирован после создания canvas)
        self.renderer = None
        
        # Создаем GUI
        self.create_ui()
        
        # Первый рендер
        self.root.after(100, self.render)
    
    def create_scenes(self):
        """Создает предустановленные сцены"""
        scenes = []
        # Сцена 1: Три куба (разнесены, чтобы не пересекались)
        cube1 = hexahedron().scale(0.75, 0.75, 0.75).translate(-2.5, 0, 0)
        cube2 = hexahedron().scale(0.75, 0.75, 0.75).translate(0, 0, -2.0)
        cube3 = hexahedron().scale(0.75, 0.75, 0.75).translate(2.5, 0, 2.0)
        scenes.append(Scene(
            "Три куба",
            [
                (cube1, (255, 100, 100)),
                (cube2, (100, 255, 100)),
                (cube3, (100, 100, 255)),
            ],
            camera_pos=[0, 3, 8],
            camera_target=[0, 0, 0]
        ))

        # Сцена 2: Платоновы тела (разнесены по X и Z)
        tetra = tetrahedron().scale(1.2, 1.2, 1.2).translate(-3.5, 0, 0)
        octa = octahedron().scale(1.2, 1.2, 1.2).translate(0, 0, -2.0)
        icosa = icosahedron().scale(1.2, 1.2, 1.2).translate(3.5, 0, 2.0)
        scenes.append(Scene(
            "Платоновы тела",
            [
                (tetra, (255, 200, 100)),
                (octa, (200, 100, 255)),
                (icosa, (100, 255, 200)),
            ],
            camera_pos=[0, 4, 10],
            camera_target=[0, 0, 0]
        ))

        # Сцена 3: Пирамида (уровни раздвинуты)
        base1 = hexahedron().scale(0.55, 0.55, 0.55).translate(-1.6, -1.0, 0.0)
        base2 = hexahedron().scale(0.55, 0.55, 0.55).translate(1.6, -1.0, 0.0)
        base3 = hexahedron().scale(0.55, 0.55, 0.55).translate(0.0, -1.0, 1.8)
        mid   = hexahedron().scale(0.55, 0.55, 0.55).translate(0.0, 0.8, 0.0)
        top   = tetrahedron().scale(0.75, 0.75, 0.75).translate(0.0, 2.6, 0.0)
        scenes.append(Scene(
            "Пирамида",
            [
                (base1, (255, 150, 150)),
                (base2, (150, 255, 150)),
                (base3, (150, 150, 255)),
                (mid, (255, 255, 150)),
                (top, (255, 150, 255)),
            ],
            camera_pos=[3, 3, 7],
            camera_target=[0, 0, 0]
        ))

        # Сцена 4: Сложное перекрытие (разнесено)
        d1 = dodecahedron().scale(0.7, 0.7, 0.7).translate(-3.0, 0, 0.0).rotate_x(30)
        c1 = hexahedron().scale(0.9, 0.9, 0.9).translate(0.0, 0, -2.5).rotate_y(45)
        o1 = octahedron().scale(1.1, 1.1, 1.1).translate(3.0, 0, 2.5).rotate_z(20)
        scenes.append(Scene(
            "Сложное перекрытие",
            [
                (d1, (255, 180, 180)),
                (c1, (180, 255, 180)),
                (o1, (180, 180, 255)),
            ],
            camera_pos=[0, 2, 9],
            camera_target=[0, 0, 0]
        ))

        # Сцена 5: Вложенные многогранники (теперь просто разнесены)
        outer = icosahedron().scale(1.5, 1.5, 1.5).rotate_x(20).translate(-4.0, 0.0, 0.0)
        middle = dodecahedron().scale(1.0, 1.0, 1.0).rotate_y(30).translate(0.0, 0.0, -2.0)
        inner = hexahedron().scale(0.6, 0.6, 0.6).rotate_z(45).translate(4.0, 0.0, 2.0)
        scenes.append(Scene(
            "Вложенные многогранники",
            [
                (outer, (255, 100, 100)),
                (middle, (100, 255, 100)),
                (inner, (100, 100, 255)),
            ],
            camera_pos=[0, 3, 10],
            camera_target=[0, 0, 0]
        ))

        return scenes
    
    def create_ui(self):
        """Создает пользовательский интерфейс"""
        # Верхняя панель управления
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Настройки Z-буфера
        zbuffer_frame = ttk.LabelFrame(top_frame, text="Алгоритм Z-буфера", padding=10)
        zbuffer_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(
            zbuffer_frame,
            text="Включить Z-буфер",
            variable=self.use_zbuffer,
            command=self.render
        ).pack(anchor=tk.W)
        
        ttk.Label(zbuffer_frame, text="(Без Z-буфера: порядок отрисовки)", 
                 font=('Arial', 8)).pack(anchor=tk.W)
        ttk.Checkbutton(
            zbuffer_frame,
            text="Отсечение нелицевых граней",
            variable=self.use_culling,
            command=self.render
        ).pack(anchor=tk.W, pady=(6,0))
        
        # Настройки проекции
        proj_frame = ttk.LabelFrame(top_frame, text="Проекция", padding=10)
        proj_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            proj_frame,
            text="Перспективная",
            value="perspective",
            variable=self.projection_mode,
            command=self.render
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            proj_frame,
            text="Параллельная (ортографическая)",
            value="parallel",
            variable=self.projection_mode,
            command=self.render
        ).pack(anchor=tk.W)
        
        # Выбор сцены
        scene_frame = ttk.LabelFrame(top_frame, text="Выбор сцены", padding=10)
        scene_frame.pack(side=tk.LEFT, padx=5)
        
        for i, scene in enumerate(self.scenes):
            ttk.Button(
                scene_frame,
                text=f"{i+1}. {scene.name}",
                command=lambda idx=i: self.change_scene(idx)
            ).pack(fill=tk.X, pady=2)
        
        # Управление камерой
        camera_frame = ttk.LabelFrame(top_frame, text="Управление камерой", padding=10)
        camera_frame.pack(side=tk.LEFT, padx=5)
        
        btn_frame1 = ttk.Frame(camera_frame)
        btn_frame1.pack()
        ttk.Button(btn_frame1, text="◄", width=3, 
                  command=lambda: self.rotate_camera(-10, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame1, text="▲", width=3,
                  command=lambda: self.rotate_camera(0, 10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame1, text="▼", width=3,
                  command=lambda: self.rotate_camera(0, -10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame1, text="►", width=3,
                  command=lambda: self.rotate_camera(10, 0)).pack(side=tk.LEFT, padx=2)
        
        btn_frame2 = ttk.Frame(camera_frame)
        btn_frame2.pack(pady=5)
        ttk.Button(btn_frame2, text="+ Ближе", width=8,
                  command=lambda: self.zoom_camera(0.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="− Дальше", width=8,
                  command=lambda: self.zoom_camera(-0.5)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(camera_frame, text="Сброс камеры", 
                  command=self.reset_camera).pack(pady=2)
        
        # Автовращение
        auto_frame = ttk.Frame(camera_frame)
        auto_frame.pack(pady=5)
        self.auto_rotate = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            auto_frame,
            text="Автовращение",
            variable=self.auto_rotate,
            command=self.toggle_auto_rotate
        ).pack()
        
        self.animation_id = None
        
        # Canvas для отрисовки
        self.canvas = tk.Canvas(self.root, bg='black', width=1000, height=700)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self.on_resize())
        
        # Метка с информацией
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.info_label = ttk.Label(
            info_frame,
            text="Z-буфер: алгоритм удаления невидимых поверхностей",
            font=('Arial', 10)
        )
        self.info_label.pack()
    
    def change_scene(self, scene_idx):
        """Меняет текущую сцену"""
        self.current_scene_idx = scene_idx
        scene = self.scenes[scene_idx]
        self.camera.reset(position=scene.camera_pos, target=scene.camera_target)
        self.render()
    
    def rotate_camera(self, delta_theta, delta_phi):
        """Вращает камеру"""
        self.camera.orbit_rotate(delta_theta, delta_phi)
        self.render()
    
    def zoom_camera(self, delta):
        """Приближает/отдаляет камеру"""
        self.camera.zoom(delta)
        self.render()
    
    def reset_camera(self):
        """Сбрасывает камеру к начальной позиции текущей сцены"""
        scene = self.scenes[self.current_scene_idx]
        self.camera.reset(position=scene.camera_pos, target=scene.camera_target)
        self.render()
    
    def toggle_auto_rotate(self):
        """Включает/выключает автовращение"""
        if self.auto_rotate.get():
            self.start_auto_rotate()
        else:
            self.stop_auto_rotate()
    
    def start_auto_rotate(self):
        """Запускает автовращение"""
        if self.animation_id is None:
            self.animate()
    
    def stop_auto_rotate(self):
        """Останавливает автовращение"""
        if self.animation_id is not None:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
    
    def animate(self):
        """Анимация автовращения"""
        if self.auto_rotate.get():
            self.camera.orbit_rotate(2, 0)
            self.render()
            self.animation_id = self.root.after(33, self.animate)
        else:
            self.animation_id = None
    
    def on_resize(self):
        """Обработка изменения размера окна"""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width > 10 and height > 10:
            # обновляем аспект и перерендерим
            self.camera.aspect = width / height
            self.render()
    
    def render(self):
        """Основной метод рендеринга"""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 10 or height < 10:
            return

        # ═══ Шаги 1-2: создаём буферы кадра и глубины ═══
        color_buffer, z_buffer = clear_buffers(width, height)
        
        # Получаем текущую сцену
        scene = self.scenes[self.current_scene_idx]
        
        # Матрица вида
        view_matrix = self.camera.get_view_matrix()
        
        # Матрица проекции
        mode = self.projection_mode.get()
        if mode == 'perspective':
            proj_matrix = self.camera.get_projection_matrix()
            use_perspective = True
        else:
            # Параллельная проекция: используем ортографическую матрицу,
            # но вместо орбитального zoom камеры меняем глобальный scale.
            proj_matrix = ortho_xy()
            use_perspective = False
            # Простейшая адаптация масштаба: чем дальше камера (orbit_radius), тем больше уменьшаем фигуры.
            # Ограничиваем коэффициент для стабильности.
            ortho_scale = 1.0 / max(0.5, self.camera.orbit_radius * 0.25)
            # Вносим дополнительное масштабирование в view (через uniform scale матрицу S).
            view_matrix = view_matrix @ np.array([
                [ortho_scale,0,0,0],
                [0,ortho_scale,0,0],
                [0,0,ortho_scale,0],
                [0,0,0,1]
            ], dtype=float)
        
        # Рендерим каждый объект, используя раздельные view и projection,
        # чтобы глубина бралась до перспективного деления.

        enable_z = self.use_zbuffer.get()

        # ═══ ФОРМАЛЬНЫЙ АЛГОРИТМ: Шаги 3-7 ═══
        # Рендерим объекты (порядок важен только если Z-буфер отключён)
        do_cull = self.use_culling.get()
        for polyhedron, color in scene.objects:
            render_polyhedron(
                polyhedron,
                view_matrix,
                proj_matrix,
                color,
                color_buffer,
                z_buffer,
                use_perspective=use_perspective,
                cull_backfaces=do_cull,
                enable_zbuffer=enable_z,
                camera=self.camera
            )
        
        # Отображаем результат
        self.display_image(color_buffer)
        
        # Обновляем информацию
        zbuffer_status = "Включен" if self.use_zbuffer.get() else "Выключен"
        proj_status = "Перспективная" if self.projection_mode.get() == 'perspective' else "Параллельная"
        cull_status = "Вкл" if self.use_culling.get() else "Выкл"
        self.info_label.config(
            text=f"Сцена: {scene.name} | Z-буфер: {zbuffer_status} | Проекция: {proj_status} | Culling: {cull_status}"
        )
    
    def display_image(self, image):
        """Отображает изображение на canvas"""
        try:
            # Создаем PhotoImage из numpy массива
            # Tkinter требует PPM формат
            height, width = image.shape[:2]
            
            # Формируем PPM данные
            ppm_header = f'P6 {width} {height} 255 '.encode()
            ppm_data = ppm_header + image.tobytes()
            
            # Создаем изображение
            photo = tk.PhotoImage(width=width, height=height, data=ppm_data, format='PPM')
            
            # Отображаем на canvas
            self.canvas.delete('all')
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            
            # Сохраняем ссылку, чтобы изображение не удалилось
            self.canvas.image = photo
            
        except Exception as e:
            print(f"Ошибка отображения: {e}")


def main():
    """Запуск приложения"""
    root = tk.Tk()
    app = ZBufferApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
