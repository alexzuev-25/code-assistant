<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1440, initial-scale=1.0">
    <meta name="filename" content="university_login.html">
    <title>Вход в систему | Учебный портал</title>
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            50: '#eff6ff',
                            100: '#dbeafe',
                            500: '#3b82f6',
                            600: '#2563eb', // Primary Blue
                            700: '#1d4ed8',
                            900: '#1e3a8a',
                        }
                    },
                    boxShadow: {
                        'card': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    }
                }
            }
        }
    </script>

    <style>
        /* Фиксация ширины для соответствия требованиям дизайна */
        body {
            background-color: #f3f4f6;
            margin: 0;
            display: flex;
            justify-content: center;
            min-height: 100vh;
        }
        
        #app-container {
            width: 1440px;
            background-color: white;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Custom Radio Button Styling */
        .role-radio:checked + div {
            border-color: #2563eb;
            background-color: #eff6ff;
            color: #1e3a8a;
        }
        
        .role-radio:checked + div .radio-dot {
            border-color: #2563eb;
            background-color: #2563eb;
        }

        .input-group:focus-within i {
            color: #2563eb;
        }
    </style>
</head>
<body>

    <div id="app-container">
        <!-- Background Image Section -->
        <div class="absolute inset-0 z-0">
            <img src="https://images.unsplash.com/photo-1541339907198-e08756dedf3f?q=80&w=2940&auto=format&fit=crop" 
                 alt="University Campus" 
                 class="w-full h-full object-cover">
            <!-- Overlay -->
            <div class="absolute inset-0 bg-brand-900/80 header-gradient"></div>
        </div>

        <!-- Main Content Area -->
        <div class="relative z-10 flex flex-col items-center justify-center w-full min-h-[900px] py-20">
            
            <!-- Login Card -->
            <div class="bg-white rounded-2xl shadow-card w-[520px] p-12 animate-fade-in">
                
                <!-- University Logo Section -->
                <div class="flex flex-col items-center justify-center mb-10">
                    <div class="w-20 h-20 bg-brand-600 rounded-full flex items-center justify-center shadow-lg mb-4 text-white">
                        <i class="fa-solid fa-graduation-cap text-4xl"></i>
                    </div>
                    <h1 class="text-2xl font-bold text-gray-800 tracking-tight text-center">Государственный Университет</h1>
                    <p class="text-gray-500 text-sm mt-1">Образовательный портал</p>
                </div>

                <!-- Form Header -->
                <div class="mb-8 text-center">
                    <h2 class="text-3xl font-bold text-gray-900">Вход в систему</h2>
                    <p class="text-gray-500 mt-2 text-base">Введите свои данные для доступа к личному кабинету</p>
                </div>

                <!-- Form Inputs -->
                <div class="space-y-6">
                    
                    <!-- Login Input -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Логин</label>
                        <div class="input-group relative rounded-md shadow-sm">
                            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                <i class="fa-regular fa-user text-gray-400 transition-colors duration-200"></i>
                            </div>
                            <input type="text" 
                                   class="block w-full pl-12 pr-4 py-3.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 text-gray-900 placeholder-gray-400 transition-all duration-200 outline-none sm:text-sm" 
                                   placeholder="Введите номер студ. билета или email">
                        </div>
                    </div>

                    <!-- Password Input -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Пароль</label>
                        <div class="input-group relative rounded-md shadow-sm">
                            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                <i class="fa-solid fa-lock text-gray-400 transition-colors duration-200"></i>
                            </div>
                            <input type="password" 
                                   class="block w-full pl-12 pr-4 py-3.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-brand-500 text-gray-900 placeholder-gray-400 transition-all duration-200 outline-none sm:text-sm" 
                                   placeholder="••••••••">
                        </div>
                    </div>

                    <!-- Role Selection -->
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-3">Роль</label>
                        <div class="grid grid-cols-3 gap-3">
                            <!-- Student -->
                            <label class="cursor-pointer group">
                                <input type="radio" name="role" class="role-radio hidden" checked>
                                <div class="border border-gray-200 rounded-lg p-3 flex flex-col items-center justify-center transition-all duration-200 hover:border-brand-300">
                                    <div class="w-4 h-4 rounded-full border border-gray-300 mb-2 flex items-center justify-center radio-dot"></div>
                                    <span class="text-sm font-medium text-gray-600 group-hover:text-brand-600">Студент</span>
                                </div>
                            </label>

                            <!-- Teacher -->
                            <label class="cursor-pointer group">
                                <input type="radio" name="role" class="role-radio hidden">
                                <div class="border border-gray-200 rounded-lg p-3 flex flex-col items-center justify-center transition-all duration-200 hover:border-brand-300">
                                    <div class="w-4 h-4 rounded-full border border-gray-300 mb-2 flex items-center justify-center radio-dot"></div>
                                    <span class="text-sm font-medium text-gray-600 group-hover:text-brand-600">Преподаватель</span>
                                </div>
                            </label>

                            <!-- Admin -->
                            <label class="cursor-pointer group">
                                <input type="radio" name="role" class="role-radio hidden">
                                <div class="border border-gray-200 rounded-lg p-3 flex flex-col items-center justify-center transition-all duration-200 hover:border-brand-300">
                                    <div class="w-4 h-4 rounded-full border border-gray-300 mb-2 flex items-center justify-center radio-dot"></div>
                                    <span class="text-sm font-medium text-gray-600 group-hover:text-brand-600">Админ</span>
                                </div>
                            </label>
                        </div>
                    </div>

                    <!-- Submit Button -->
                    <div class="pt-4">
                        <button type="button" class="w-full flex justify-center py-4 px-4 border border-transparent rounded-xl shadow-sm text-sm font-bold text-white bg-brand-600 hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-500 transition-all duration-200 transform hover:-translate-y-0.5 uppercase tracking-wide">
                            ВОЙТИ
                        </button>
                    </div>

                    <!-- Forgot Password Link -->
                    <div class="text-center pt-2">
                        <a href="#" class="font-medium text-brand-600 hover:text-brand-500 hover:underline transition-all duration-150 text-sm">
                            Забыли пароль?
                        </a>
                    </div>

                </div>
            </div>

            <!-- Footer Info -->
            <div class="mt-8 text-center text-white/70 text-sm">
                <p>&copy; 2023 Государственный Университет. Все права защищены.</p>
                <div class="flex gap-4 justify-center mt-2">
                    <a href="#" class="hover:text-white transition-colors">Помощь</a>
                    <span>|</span>
                    <a href="#" class="hover:text-white transition-colors">Конфиденциальность</a>
                </div>
            </div>
        </div>
    </div>

</body>
</html>