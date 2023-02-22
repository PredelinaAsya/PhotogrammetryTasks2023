// Убедитесь что название PR соответствует шаблону:
// Task02 <Имя> <Фамилия> <ВУЗ>

// Проверьте что обе ветки PR - task02 (отправляемая из вашего форкнутого репозитория и та в которую вы отправляете PR)

# Перечислите идеи и коротко обозначьте мысли которые у вас возникали по мере выполнения задания, в частности попробуйте ответить на вопросы:

1) Зачем фильтровать матчи, если потом мы запускаем устойчивый к выбросам RANSAC и отфильтровываем шумные сопоставления?

2) Cluster filtering довольно хорошо работает и без Ratio test. Однако, если оставить только Cluster filtering, некоторые тесты начнут падать. Почему так происходит? В каких случаях наоборот, не хватает Ratio test и необходима дополнительная фильтрация?

3) С какой проблемой можно столкнуться при приравнивании единице элемента H33 матрицы гомографии? Как ее решить?

4) Какой подвох таится в попытке склеивать большие панорамы и ортофото методом, реализованным в данной домашке? (Для интуиции можно посмотреть на результат склейки, когда за корень взята какая-нибудь другая картинка)

5) Как можно автоматически построить граф для построения панорамы, чтобы на вход метод принимал только список картинок?

6) Если с вашей реализацией SIFT пройти тесты не получилось, напишите (если пробовали дебажить), где, как вам кажется, проблема и как вы пробовали ее решать.

7) Если есть, фидбек по заданию: какая часть больше всего понравилась, где-то слишком сложно/просто (что именно), где-то слишком мало ссылок и тд.


// Создайте PR.
// Дождитесь отработки Travis CI, после чего нажмите на зеленую галочку -> Details -> The build -> скопируйте весь лог тестирования.
// Откройте PR на редактирование (сверху справа три точки->Edit) и добавьте сюда скопированный лог тестирования внутри тега <pre> для сохранения форматирования и под спойлером для компактности и удобства:

<details><summary>Travis CI</summary><p>

<pre>
$ ./build/test_sift
$ ./build/test_matching
Running main() from /home/runner/work/PhotogrammetryTasks2023/PhotogrammetryTasks2023/libs/3rdparty/libgtest/googletest/src/gtest_main.cc
[==========] Running 22 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 22 tests from SIFT
[ RUN      ] SIFT.MovedTheSameImage
[ORB_OCV] Points detected: 500 -> 500 (in 0.021269 sec)
...
[       OK ] SIFT.HerzJesu19RotateM40 (7730 ms)
[----------] 22 tests from SIFT (12918 ms total)
[----------] Global test environment tear-down
[==========] 22 tests from 1 test suite ran. (12918 ms total)
[  PASSED  ] 22 tests.
...
</pre>

</p></details>