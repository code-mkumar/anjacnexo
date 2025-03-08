import sqlite3
import os
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/university.db"))
conn=sqlite3.connect(db_path)
#conn=sqlite3.connect("university.db")
mycursor=conn.cursor()

#mycursor.execute("select * from student_details")
# query ='''SELECT 
#             sd.id AS student_id,
#             sd.name AS student_name,
#             sd.class AS class,
#             s.name AS subject_name,
#             sm.quiz1,
#             sm.quiz2,
#             sm.quiz3,
#             sm.assignment1,
#             sm.assignment2,
#             sm.internal1,
#             sm.internal2,
#             sm.internal3
#         FROM 
#             student_details sd
#         JOIN 
#             student_mark_details sm ON sd.id = sm.student_id
#         JOIN 
#             subject s ON sm.subject_id = s.id
#         WHERE 
#             sd.department_id = 'PGCS' AND sd.class = 'II' AND s.id = '23PCSC412'
#         limit 5;
# '''
#query='''SELECT DISTINCT s.name, smd.quiz1, smd.quiz2, smd.quiz3, smd.assignment1, smd.assignment2, smd.internal1, smd.internal2, smd.internal3 FROM student_details s JOIN subject sub ON s.id = sub.department_id JOIN timetable tm ON s.id = tm.student_id AND sub.name = 'Compiler Design' AND tm.class = 'II';'''
query=''' update student_mark_details set internal3=25.90 where student_id="23UCS031"'''
query='''select * from student_mark_details;'''
mycursor.execute(query)
# cols = []
# for  desc in mycursor.description:
#     cols.append(desc[0])
# print("column names:",cols)
# res=mycursor.fetchall()
# # for i in res:
# #     print(i)

# for row in res:
#     row_dict=dict(zip(cols,row))
#     print(row_dict)

# mycursor.execute("select * from student_mark_details")
res=mycursor.fetchall()
print(res)