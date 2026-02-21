FRAME_IMAGES = {
    "Cat Eye": "spectacles/cateye.png",
    "Rectangle": "/spectacles/rectangle.jpg",
    "Wayfare": "/spectacles/wayfarer.jpg",
    "Square": "/spectacles/square.png",
    "Browline": "/spectacles/browline.jpg",
    "Round": "/spectacles/round.jpg",
    "Aviators": "/spectacles/aviator.jpg",
    "Geometric": "/spectacles/geometric.jpg",
    "Oval": "/spectacles/oval.jpg"
}

# Senarai Frame (Macam asal)
RECOMMENDATIONS = {
    "heart": [
        "Cat Eye", "Rectangle", "Wayfare", "Square", "Browline"
    ],
    "oblong": [
        "Round", "Cat Eye", "Wayfare", "Aviators", "Geometric"
    ],
    "oval": [
        "Round", "Cat Eye", "Rectangle", "Wayfare", "Square", 
        "Aviators", "Geometric", "Browline", "Oval"
    ],   
    "round": [
        "Rectangle", "Wayfare", "Square", "Geometric", "Browline"
    ],
    "square": [
        "Round", "Cat Eye", "Wayfare", "Aviators", "Geometric", "Oval"
    ]
}

# Senarai Traits (Baru ditambah)
TRAITS = {
    "heart": "Broad forehead and high cheekbones narrowing down to a sharp, pointed chin.",
    "oblong": "Face is longer than it is wide, with straight cheeklines and a rounded chin.",
    "oval": "Balanced proportions. The forehead is slightly wider than the chin, with soft, curved jawlines.",
    "round": "Face width and length are roughly equal. Full cheeks with soft angles and no sharp jawline.",
    "square": "Strong, sharp jawline. The forehead, cheekbones, and jaw are all about the same width."
}

def get_recommendations(face_shape):
    if not face_shape:
        return None
    
    key = face_shape.lower()
    
    # Ambil list nama frame
    frame_names = RECOMMENDATIONS.get(key, ["Classic", "Wayfare"])
    
    # Ambil traits
    traits = TRAITS.get(key, "Unique face features.")
    
    # 3. PROSES DATA: Tukar list nama jadi list object (Nama + Gambar)
    structured_frames = []
    
    for name in frame_names:
        # Cari filename, kalau tak ada guna 
        image_file = FRAME_IMAGES.get(name, "src\spectacles\geometric.jpg")
        
        structured_frames.append({
            "name": name,
            "image": image_file
        })

    # Return structure baru
    return {
        "frames": structured_frames, # Ini sekarang list of objects
        "traits": traits
    }