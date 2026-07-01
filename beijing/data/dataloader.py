# 创建新文件: dataloader.py
class GeoBounds:
    def __new__(cls, area='scs'):
        if area == 'wpo':
            print("Using WPO geo bounds")
            return {
                'lat_min': 4.0720,
                'lat_max': 44.3000,
                'lon_min': 90.7000,
                'lon_max': 169.4000,
                'lat_trg_min': 4.7000,
                'lat_trg_max': 44.3273,
                'lon_trg_min': 91.0100,
                'lon_trg_max': 166.6150
            }
        else:
            print("Using SCS geo bounds")
            return {
                'lat_min': 5.7,
                'lat_max': 25.271,
                'lon_min': 102.4,
                'lon_max': 129.669,
                'lat_trg_min': 5.8469,
                'lat_trg_max': 20.9486,
                'lon_trg_min': 105.0250,
                'lon_trg_max': 117.9000
            }
