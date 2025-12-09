<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="filename" content="dashboard_ml.html">
    <title>Управление моделями машинного обучения</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>

    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#2563EB',
                        secondary: '#475569',
                        success: '#10B981',
                        warning: '#F59E0B',
                        danger: '#EF4444',
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #F3F4F6;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
        }
        .dashboard-container {
            width: 1440px;
            /* Height adapts to content, min-height for visuals */
            min-height: 100vh; 
            background-color: #ffffff;
            box-shadow: 0 0 20px rgba(0,0,0,0.05);
            padding: 2rem;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .model-card {
            transition: all 0.2s ease;
        }
        .model-card:hover {
            border-color: #BFDBFE;
            background-color: #F8FAFC;
        }
    </style>
</head>
<body class="text-slate-800">

    <div class="dashboard-container">
        
        <!-- Header -->
        <header class="mb-8 border-b border-gray-200 pb-6 flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold text-slate-900 mb-2">Управление моделями машинного обучения</h1>
                <p class="text-slate-500">Панель администратора / v2.4.0</p>
            </div>
            <div class="flex items-center gap-4">
                <div class="text-right">
                    <div class="font-medium">Admin User</div>
                    <div class="text-xs text-slate-500">Super Admin</div>
                </div>
                <div class="h-10 w-10 bg-primary rounded-full flex items-center justify-center text-white">
                    <i class="fa-solid fa-user"></i>
                </div>
            </div>
        </header>

        <!-- Main Grid Layout -->
        <div class="grid grid-cols-12 gap-8">
            
            <!-- LEFT COLUMN (Models & Stats) -->
            <div class="col-span-8 flex flex-col gap-8">
                
                <!-- 1. Model List Section -->
                <div class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                    <div class="bg-gray-50 px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                        <h2 class="text-lg font-semibold text-slate-800">
                            <i class="fa-solid fa-layer-group mr-2 text-primary"></i>
                            Доступные модели
                        </h2>
                        <span class="bg-blue-100 text-primary text-xs font-bold px-2 py-1 rounded">3 активных</span>
                    </div>
                    
                    <div class="p-6 space-y-4">
                        <!-- Model 1 -->
                        <div class="model-card border border-gray-200 rounded-lg p-5 flex justify-between items-center cursor-pointer relative overflow-hidden group">
                            <div class="absolute left-0 top-0 bottom-0 w-1 bg-success group-hover:w-1.5 transition-all"></div>
                            <div>
                                <div class="flex items-center gap-3 mb-1">
                                    <h3 class="font-bold text-lg">CodeGPT v1.0</h3>
                                    <span class="bg-green-100 text-success text-xs px-2 py-0.5 rounded-full font-medium">87% точность</span>
                                </div>
                                <div class="flex items-center text-sm text-slate-500">
                                    <span class="status-dot bg-success"></span>
                                    <span class="text-success font-medium">Активна</span>
                                    <span class="mx-2 text-gray-300">|</span>
                                    <span>Deployed: 2 days ago</span>
                                </div>
                            </div>
                            <div class="text-slate-400 group-hover:text-primary transition-colors">
                                <i class="fa-solid fa-chevron-right"></i>
                            </div>
                        </div>

                        <!-- Model 2 -->
                        <div class="model-card border border-gray-200 rounded-lg p-5 flex justify-between items-center cursor-pointer relative overflow-hidden group">
                            <div class="absolute left-0 top-0 bottom-0 w-1 bg-warning group-hover:w-1.5 transition-all"></div>
                            <div>
                                <div class="flex items-center gap-3 mb-1">
                                    <h3 class="font-bold text-lg">CodeBERT v2.1</h3>
                                    <span class="bg-yellow-100 text-orange-600 text-xs px-2 py-0.5 rounded-full font-medium">82% точность</span>
                                </div>
                                <div class="flex items-center text-sm text-slate-500">
                                    <span class="status-dot bg-warning"></span>
                                    <span class="text-warning font-medium">Ожидает проверки</span>
                                    <span class="mx-2 text-gray-300">|</span>
                                    <span>Training finished: 1 hour ago</span>
                                </div>
                            </div>
                            <div>
                                <button class="text-sm border border-gray-300 px-3 py-1 rounded hover:bg-gray-50 text-slate-600">
                                    Логи
                                </button>
                            </div>
                        </div>

                        <!-- Model 3 -->
                        <div class="model-card border border-gray-200 rounded-lg p-5 flex justify-between items-center cursor-pointer relative overflow-hidden group">
                            <div class="absolute left-0 top-0 bottom-0 w-1 bg-danger group-hover:w-1.5 transition-all"></div>
                            <div>
                                <div class="flex items-center gap-3 mb-1">
                                    <h3 class="font-bold text-lg">CustomModel v1.5</h3>
                                    <span class="bg-red-100 text-danger text-xs px-2 py-0.5 rounded-full font-medium">91% точность</span>
                                </div>
                                <div class="flex items-center text-sm text-slate-500">
                                    <span class="status-dot bg-danger"></span>
                                    <span class="text-danger font-medium">Ошибка</span>
                                    <span class="mx-2 text-gray-300">|</span>
                                    <span>Error: OOM Exception</span>
                                </div>
                            </div>
                            <div>
                                <button class="text-sm border border-red-200 text-red-600 px-3 py-1 rounded hover:bg-red-50">
                                    Детали
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 3. Statistics Section -->
                <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                    <h2 class="text-lg font-semibold text-slate-800 mb-6 flex items-center">
                        <i class="fa-solid fa-chart-line mr-2 text-primary"></i> 
                        Статистика использования
                    </h2>
                    
                    <div class="grid grid-cols-3 gap-6 mb-6">
                        <!-- Stat Card 1 -->
                        <div class="bg-blue-50 p-4 rounded-lg border border-blue-100">
                            <p class="text-slate-500 text-sm mb-1">Запросов сегодня</p>
                            <div class="text-2xl font-bold text-slate-800">1,245</div>
                            <div class="text-xs text-green-600 mt-1"><i class="fa-solid fa-arrow-up"></i> +12% от вчера</div>
                        </div>
                        <!-- Stat Card 2 -->
                        <div class="bg-indigo-50 p-4 rounded-lg border border-indigo-100">
                            <p class="text-slate-500 text-sm mb-1">Среднее время отклика</p>
                            <div class="text-2xl font-bold text-slate-800">4.2с</div>
                            <div class="text-xs text-slate-500 mt-1">Оптимально < 5.0с</div>
                        </div>
                        <!-- Stat Card 3 -->
                        <div class="bg-emerald-50 p-4 rounded-lg border border-emerald-100">
                            <p class="text-slate-500 text-sm mb-1">Доступность API</p>
                            <div class="text-2xl font-bold text-emerald-600">99.8%</div>
                            <div class="text-xs text-green-600 mt-1">Все системы в норме</div>
                        </div>
                    </div>

                    <!-- EChart Container -->
                    <div id="statsChart" class="w-full h-64 rounded-lg overflow-hidden"></div>
                </div>

            </div>

            <!-- RIGHT COLUMN (Actions & Upload) -->
            <div class="col-span-4 flex flex-col gap-8">
                
                <!-- 2. Actions Panel -->
                <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 h-fit">
                    <h2 class="text-lg font-semibold text-slate-800 mb-4 border-b pb-2">Действия</h2>
                    
                    <div class="space-y-3">
                        <button class="w-full bg-primary hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 shadow-sm">
                            <i class="fa-solid fa-plus"></i> Загрузить новую модель
                        </button>
                        
                        <button class="w-full bg-white border border-gray-300 hover:bg-gray-50 text-slate-700 font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2">
                            <i class="fa-solid fa-rotate"></i> Дообучить модель
                        </button>
                        
                        <button class="w-full bg-white border border-gray-300 hover:bg-gray-50 text-slate-700 font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2">
                            <i class="fa-solid fa-file-export"></i> Экспорт модели
                        </button>
                        
                        <button class="w-full bg-white border border-red-200 hover:bg-red-50 text-red-600 font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mt-4">
                            <i class="fa-solid fa-trash-can"></i> Удалить модель
                        </button>
                    </div>
                    
                    <div class="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-100">
                        <div class="flex gap-2">
                            <i class="fa-solid fa-triangle-exclamation text-yellow-500 mt-1"></i>
                            <div>
                                <h4 class="text-sm font-bold text-yellow-800">Внимание</h4>
                                <p class="text-xs text-yellow-700 mt-1">Обслуживание сервера запланировано на 22:00.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 4. Data Upload Section -->
                <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 flex-grow">
                    <h2 class="text-lg font-semibold text-slate-800 mb-4 border-b pb-2">Загрузка данных</h2>
                    
                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-gray-50 hover:bg-blue-50 hover:border-blue-300 transition-colors cursor-pointer group">
                        <div class="w-12 h-12 bg-white rounded-full mx-auto flex items-center justify-center shadow-sm mb-3 group-hover:scale-110 transition-transform">
                            <i class="fa-solid fa-cloud-arrow-up text-primary text-xl"></i>
                        </div>
                        <p class="text-sm font-medium text-slate-700">Выберите файл .csv</p>
                        <p class="text-xs text-slate-400 mt-1">или перетащите сюда</p>
                        <p class="text-xs text-slate-400 mt-4">Макс. размер: 500MB</p>
                    </div>

                    <div class="mt-4">
                        <label class="block text-sm font-medium text-slate-700 mb-1">Имя датасета</label>
                        <input type="text" placeholder="Введите название..." class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 text-sm mb-4">
                        
                        <button class="w-full bg-slate-800 hover:bg-slate-900 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2">
                            <i class="fa-solid fa-database"></i> Загрузить датасет
                        </button>
                    </div>
                </div>

            </div>
        </div>

    </div>

    <script>
        // Init Chart
        const chartDom = document.getElementById('statsChart');
        const myChart = echarts.init(chartDom);
        const option = {
            tooltip: {
                trigger: 'axis'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                top: '10%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                axisLine: { lineStyle: { color: '#94a3b8' } }
            },
            yAxis: {
                type: 'value',
                axisLine: { show: false },
                splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } }
            },
            series: [
                {
                    name: 'Requests',
                    type: 'line',
                    smooth: true,
                    lineStyle: {
                        color: '#3B82F6',
                        width: 3
                    },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                            { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
                        ])
                    },
                    showSymbol: false,
                    data: [120, 132, 450, 932, 1100, 850, 600]
                }
            ]
        };

        myChart.setOption(option);

        // Make chart responsive
        window.addEventListener('resize', function() {
            myChart.resize();
        });
    </script>
</body>
</html>