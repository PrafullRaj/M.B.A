Final Project Report

UNIVERSITY ANALYTICS SYSTEM

 



PREPARED BY:
ASSOCIATE ID	ASSOCIATE NAME
2387487	Prafull Raj










GUIDED BY:
Mentor & Trainer                                               Coach
Kavin Kumar G	                                                Akash Nair
Associate	                                                          CPO L&D GenC Training
AIA - Cloud Data Integration	                           Contractor – GenC
CONTENTS

❖	Introduction                                                                                      
❖	Purpose of the Project					                       
❖	About the Software System					             
❖	Scope of the System        							   
❖	Architecture Diagram	   						   
❖	Flowchart                                                                                           
❖	Functional Requirement 1: Student Dimension    	
❖	Functional Requirement 2: College Dimension			   
❖	Functional Requirement 3: Faculty Dimension                                                                                                            
❖	Functional Requirement 4: Department Dimension     
❖	Functional Requirement 5: Course Dimension
❖	Functional Requirement 6: Time Dimension
❖	Functional Requirement 7: Result Staging
❖	Functional Requirement 8: Result Fact
❖	Functional Requirement 9: Result Aggregate
❖	Integrated Taskflow
❖	Integrated Task Details
❖	Design Constraints
❖	Conclusion                             






ABBREVIATIONS

TERM
	DESCRIPTION
ETL
	Extraction, Transformation and Loading / Extract, Transform and Load

SRC
	Source
TGT
	Target
SQ
	Source Qualifier
WF
	Workflow
EXP
	Expression
SEQ
	Sequence Generator
LKP
	Lookup
AGG
	Aggregator
FIL
	Filter
RTR
	Router
SRT 	Sorter
UN	Union
JNR	Joiner


LIST OF TABLES
Table 1: Student Dimension Source Description
Table 2: Student Target Description
Table 3: Student Dimension Transformations
Table 4: College Dimension Source Description
Table 5: College Target Description
Table 6: College Dimension Transformations
Table 7: Faculty Dimension Source Description
Table 8: Faculty Target Description
Table 9: Faculty Dimension Transformations
Table 10: Department Dimension Source Description
Table 11: Department Target Description
Table 12: Department Transformations
Table 13: Course Dimension Source Description
Table 14: Course Target Description
Table 15: Course Transformations
Table 16: Time Dimension Source Description
Table 17: Time Target Description
Table 18: Time Transformations
Table 19: Result Staging Dimension Source Description
Table 20: Result Staging Target Description 
Table 21: Result Staging Transformations
Table 22: Result Fact Dimension Source Description
Table 23: Result Fact Target Description
Table 24: Result Fact Transformations
Table 25: Result Aggregate Student Dimension Source Description
Table 26: Result Aggregate Target Description
Table 27: Result Aggregate Transformations
Table 28: Design Constraints


LIST OF FIGURES
•	Fig. 1: Architecture Diagram
•	Fig. 2: Flow Chart
•	Fig. 3: Mapping Student
•	Fig. 4: Sql Table Student 
•	Fig. 5: Task Monitor Student
•	Fig. 6: Mapping College
•	Fig. 7: Sql Table College
•	Fig. 8: Task Monitor College
•	Fig.9: Mapping Faculty
•	Fig. 10: Sql Table Faculty
•	Fig. 11: Task Monitor Faculty
•	Fig. 12: Mapping Department
•	Fig. 13: SQL Table Department
•	Fig. 14: Task Monitor Department
•	Fig. 15: Mapping Course
•	Fig. 16: SQL Table Course
•	Fig. 17: Task Monitor Course
•	Fig. 18: Mapping time
•	Fig. 19: SQL Table Time
•	Fig. 20: Task Monitor Time
•	Fig. 21: Mapping Staging
•	Fig. 22: SQL Table Staging
•	Fig. 23: Task Monitor Staging
•	Fig. 24: Mapping Fact
•	Fig. 25: SQL Table Fact
•	Fig 26: Task Monitor
•	Fig. 27: Mapping Agg
•	Fig. 28: SQL Table Agg
•	Fig. 29:Task Monitor  Agg
•	Fig. 30: Complete Task Flow
•	Fig. 30: Complete Task Flow
•	Fig 31: Task Monitor






Introduction
The University Analytics System project is designed to optimize and simplify academic data management in a university environment. By consolidating data from various dimension tables—such as Student, Department, Time, Faculty, Course, and College—it creates a unified Result table that offers a comprehensive view of essential information. This integration facilitates efficient data processing and analysis, empowering informed decision-making and enhancing institutional management practices.
This project is significant as it addresses the need for a centralized analytical system in academic institutions, ensuring data consistency, accessibility, and enhanced decision-making capabilities. The introduction outlines the scope and structure of the project, highlighting its contribution to improving operational efficiency and educational outcomes.

Purpose of the Project
The University Analytics System project is designed to create a centralized and efficient platform for managing and analyzing academic data within a university setting. By consolidating information from various dimension tables—such as Student, Department, Time, Faculty, Course, and College—the system strives to deliver precise and actionable insights that enhance decision-making and streamline institutional operations. Its goal is to improve the accessibility, consistency, and usability of academic data for administrators, faculty, and other university stakeholders.

About the Software System
The University’s sample data is loaded using the tool. The data from the source system is supplied as a flat file that is to be imported into a relational table in a database which is in a data warehouse. The data that has been entered into the data warehouse will be used to make decisions. The following sections will illustrate the different types of transformations that was performed on the modules.

Modules present in the project
1.	Student Dimension
2.	College Dimension
3.	Time Dimension
4.	Department Dimension
5.	Course Dimension
6.	Faculty Dimension
7.	Result Staging
8.	Result Fact
9.	Result Aggregate

Scope of the System
The scope of the system is to create a University Analytics System Data Warehouse and populate the tables present in them.

Architecture Diagram
The Logical Architecture defines the Processes (the activities and functions) that are required to provide the required User Services. Many different Processes must work together and share information to provide a User Service. The processes can be implemented via software, hardware, or firmware. Logical Architecture is independent of technologies and implementations.
University Analytics system has 6-dimensional table, 1 staging table and 1 fact table.

 
Fig. 1: Architecture Diagram




Flow Chart

 




      Fig. 2: Flow Chart

Functional Requirement 1
STUDENT DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_Student.txt
	This is a delimited file with 12 input fields. 
 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_STUDENT	It is a SCD type 2 table maintaining the created and end date	SQL Server Database	 

Table 2: CDW_UAS_D_STUDENT Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
STUD_ID	System generated based on Student SSN. 
Insert for new Student/ expire the old record and 
insert the updated record for existing Students	Sequence Generator	
STUD_F_NAME	Convert the Name to Title Case	Expression	InitCap()


STUD_M_NAME	Convert the middle name in lower case	Expression	Lower()

STUD_L_NAME	Convert the Last Name in Title Case	Expression	InitCap()

STUD_ROLLNO	Direct move ( Abort the session if Roll Number is invalid. Convert to number )	Expression	IIF()
STUD_STREET	Concatenate Apartment no and Street name of Student's Residence with comma as a separator	Expression	IIF()
STUD_CITY	Direct Move		Concat()
STUD_STATE	Direct Move		
STUD_COUNTRY	Direct move		
STUD_ZIP	Direct move ( Convert to number )		
STUD_PHONE	Change the format to xxx-xxx-xxxx		ToInteger()
STUD_EMAIL	Direct move		Substring()
CREATED_DATE	SYSDATE		
END_DATE	Null on insert / SYSDATE - (1Sec) on update.		SYSDATE

Table 3: CDW_UAS_D_STUDENT Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.
 
                                                          Fig. 3: Mapping Student  
TARGET POST EXECUTION:
 

                                                          Fig. 4: Sql Table Student 










TASK MONITOR
 
Fig. 5: Task Monitor Student





















Functional Requirement 2
COLLEGE DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_COLLEGE.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_COLLEGE	It is a SCD type 1 table 	SQL Server Database	 

Table 2: CDW_UAS_D_COLLEGE Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
COLLEGE_CODE	Insert if new College/update the entire record for existing College	Sequence Generator	-
COLLEGE_NAME	Direct move 	-	-
COLLEGE_STREET	Direct move 	-	-
COLLEGE_CITY	Direct move 	-	-
COLLEGE_STATE	Direct move 	-	-
COLLEGE_ZIP	If the source value is null load default value else Direct move	Expression	IIF, ISNULL
COLLEGE_PHONE	Change the format of phone number to (XXX)XXX-XXXX	Expression	SUBSTR

Table 3: CDW_UAS_D_COLLEGE Transformations
MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.
    
  					Fig. 6: Mapping College
 TARGET POST EXECUTION:

 
                                                          Fig. 7: Sql Table College

TASK MONITOR
 
				Fig. 8: Task Monitor College





















Functional Requirement 3
FACULTY DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_FACULTY.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_FACULTY	It is a SCD type 1 table	SQL Server Database	 

Table 2: CDW_UAS_D_FACULTY Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
FACULTY_CODE	Insert if new College/update the entire record for existing College	Sequence Generator	-
FACULTY_NAME	Direct move 	-	-
FACULTY_DESIGNATION	Direct move 	-	-
FACULTY_QUALIFICATION	Direct move 	-	-

Table 3: CDW_UAS_D_FACULTY Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

 
                                                          Fig.9: Mapping Faculty


                                                            
TARGET POST EXECUTION:

 
                                                          Fig. 10: Sql Table Faculty
TASK MONITOR
 
				Fig. 11: Task Monitor Faculty




















Functional Requirement 4
DEPARTMENT DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_DEPARTMENT.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_DEPARTMENT	It is a SCD type 1 table 	SQL Server Database	 

Table 2: CDW_UAS_D_ DEPARTMENT Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
DEPARTMENT_ID	System generated based on  Department No.Insert for new  Department/update the entire record for existing  Departments	Sequence Generator	-
DEPARTMENT_NAME	If a percentage symbol is present in the name of Department remove and load else direct move	Expression	IIF, INSTR
DEPARTMENT_NO	Direct move(Abort the session if Number is invalid)	-	-
DEPARTMENT_PHONE	Standardize the phone number to XXX-XXX-XXXX	-	-
DEPARTMENT_HEAD	Direct move	-	-

Table 3: CDW_UAS_D_DEPARTMENT Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

 
                                                          Fig. 12: Mapping Department
                                                  TARGET POST EXECUTION:
 
                                                         Fig. 13: SQL Table Department
TASK MONITOR 
 
				Fig. 14: Task Monitor Department




















Functional Requirement 5
COURSE DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_COURSE.txt
	This is a delimited file with 12 input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_COURSE	It is a SCD type 1 table 	SQL Server Database	 

Table 2: CDW_UAS_D_ COURSE Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
COURSE_CODE	Insert for new Course/update the entire record for existing Course( Abort the session if Code is invalid )	SEQUENCE GENERATOR	-
COURSE_NAME	Trim the trailing spaces and load the data	Expression	LTRIM, RTRIM
DEPARTMENT_ID	Join with Department table, based on No from Department Table
and load the corresponding Department ID	

Joiner	

-
MARK_REQ	Direct move	-	-

Table 3: CDW_UAS_D_ COURSE Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

 
                                                          Fig. 15: Mapping Course

                                                       TARGET POST EXECUTION:

 
                                                          Fig. 16: SQL Table Course
TASK MONITOR
 
					Fig. 17: Task Monitor Course




















Functional Requirement 6
TIME DIMENSION
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_TIME.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_TIME	Static Dimension	SQL Server Database	 

Table 2: CDW_UAS_D_ TIME Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
TIMEID	Direct move	-	-
DAY	Get the date part of TIME_ID	Expression	TO_INTEGER, SUBSTR
MONTH	Get the Month part of TIME_ID	Expression	TO_INTEGER, SUBSTR
QUARTER	Calculate using the MONTH column.	Expression	IIF, TO_CHAR,TO_INTEGER,
SUBSTR,CEIL

YEAR	Get the year part of TIME_ID	Expression	-

Table 3: CDW_UAS_D_STUDENT Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

 
                                                          Fig. 18: Mapping time
                                                            
TARGET POST EXECUTION:

 
                                                          Fig. 19: SQL Table Time
TASK MONITOR 
 
					Fig 20: Task Monitor Time




















Functional Requirement 7
RESULT STAGING
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_STG_RESULT_DL.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_D_ STG_RESULT	It is a Staging table	SQL Server Database	 

Table 2: CDW_UAS_STG_RESULT_DL Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
CDW_P_RESULT_DSET_KEY	seq NUMBER	Sequence Generator	-
RESULT_F_PERIOD_KEY	Look up from Time dim table and load the TIMEID	Lookup	-
RESULT_F_PERIOD_KEY	Look up from Time dim table and load the TIMEID	Lookup	-
RESULT_F_PERIOD_KEY	Look up from Time dim table and load the TIMEID	Lookup	-
RESULT_F_Student_KEY	look up from Student dim table. 	Lookup	-
RESULT_F_DEPARTMENT_KEY	look up from Department dim table	Lookup	-
RESULT_F_COLLEGE_CODE	Look up College Code using College name from the College Dimension Table	Lookup	-
RESULT_F_COLLEGE_NAME	Direct move	-	-
RESULT_F_COURSE_CODE	Look up Course Code using Course Name from the Course Dimension Table	Lookup	-
RESULT_F_COURSE_NAME	Direct move	-	-
RESULT_F_FACULTY_NO	Look up Faculty Code using Faculty Name from the Faculty Dimension Table	Lookup	-
RESULT_MARKS	Direct move from the file/check for decimal values	Expression	IIF,ISNULL
RESULT_GRADE	Need to be calculated from the Course table data (If the Result Marks is greater than the Marks Required then PASS else FAIL).	Expression	-
CREATED_DATE	sysdate	Expression	SYSDATE

Table 3: CDW_UAS_STG_RESULT_DL Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

                                                           Fig. 21: Mapping Staging


                                                            
TARGET POST EXECUTION:

                                                           Fig. 22: SQL Table Staging



TASK MONITOR 
 
				Fig 23: Task Monitor Staging



















Functional Requirement 8
RESULT FACT TABLE
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_F_RESULT.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_F_RESULT	It is a Fact Table	SQL Server Database	 

Table 2: CDW_UAS_F_RESULT Target Description

TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
CDW_XYZ_F_RESULT_DSET_KEY	Direct Move	-	-
RESULT_F_PERIOD_KEY	Direct Move (Convert to date datatype)	Expression	-
RESULT_F_STUDENT_KEY	Direct Move	-	-
RESULT_F_DEPARTMENT_KEY	Direct Move	-	-
RESULT_F_COLLEGE_CODE	Direct Move	-	-
RESULT_F_COLLEGE_NAME	Direct Move	-	-
RESULT_F_COURSE_CODE	Direct Move	-	-
RESULT_F_COURSE_NAME	Direct Move	-	-
RESULT_MARKS	Direct Move	-	-
RESULT_GRADE	Direct Move	-	-
CREATED_DATE	Direct Move	Expression	sysdate
CREATED_BY	Direct Move	Expression	$CurrentMappingName

Table 3: CDW_UAS_F_RESULT Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

                                                           Fig. 24: Mapping Fact


                                                            
TARGET POST EXECUTION:

                                                           Fig. 25: SQL Table Fact

TASK MONITOR 
 
				Fig 26: Task Monitor 
Functional Requirement 9
RESULT AGGREGATE TABLE
SOURCE
The first functional requirement is to load the source file provided into the Student Table 

SOURCE FILE NAME	DESCRIPTION	SOURCE FILE
CDW_UAS_AGG_F_RESULT.txt
	This is a delimited file with input fields.	 

                                              Table 1: Student Source Description

TARGET 
The target table has been generated by creating a schema from the data provided in the requirement document.

TARGET NAME	DESCRIPTION	TARGET TYPE	TARGET FILE
CDW_UAS_F_AGG_DATA	It is an aggregate table	SQL Server Database	 

Table 2: CDW_UAS_AGG_F_RESULT Target Description




TRANSFORMATIONS:
Column Name	Mapping Logic	Transformation Name	Function Used
CDW_RESULT_AGG_DSET_KEY	Sequence generated number	Sequence Generator	-
COLLEGE_CODE	College code for the current reporting period sold in each College 	-	-
COLLEGE_NAME	College name for the corresponding College	-	-
COURSE_CODE	Course code for the current reporting period evaluated in each College 	-	-
COURSE_NAME	Course name for the corresponding Course sold	-	-
TOTAL PASS	Count of the total students passes in a particular Course evaluated in each College for the current reporting period. 	Aggregator	count
MAXIMUM_MARK	Maximum mark scored in a particular Course in each college for the current reporting period. 	Aggregator	Max
CREATED_DATE	Sysdate	Aggregator	Sysdate

Table 3: CDW_UAS_AGG_F_RESULT Transformations

MAPPING
The figure below illustrates the data flow from the source to the target table, applying the necessary transformations to meet the business requirements.

                                                           Fig. 27: Mapping Agg


                                                            
TARGET POST EXECUTION:

                                                          Fig. 28: SQL Table Agg


TASK MONITOR 

 
				Fig 29	Task Monitor  Agg

INTEGRATED TASKFLOW
 
				Fig. 30: Complete Task Flow
 
					Fig 31: Task Monitor







DESIGN CONSTRAINTS
CONSTRAINT	SUCCESS	IMPLEMENTATION
Student table should get loaded only when a valid No exists in the source file.	Yes	Same column name (No) having matching values in both source and target tables.
An item record getting loaded into the fact table should have a corresponding entry in the item dimension table.	Yes	COURSE_CODE loaded to the Fact Table.
A customer record getting loaded into the fact table should have a corresponding entry in the customer dimension table.	Yes	DEPARTMENT_ID loaded to the Fact Table.
The ETL data flow schedule should accommodate the order of dimension table loading before the fact table load.	Yes	Scheduled the workflows of dimensions in parallel, loading the fact table only when all dimensions are succeeded.
The customer-item fact table should calculate the total price based on the number of items and item price.	Yes	Implemented calculation logic in the transformation step.
The phone number format should be standardized to xxx-xxx-xxxx.	Yes	Applied transformation logic to format phone numbers.
The address fields should be concatenated with a comma separator.	Yes	Used concatenation logic in the transformation step.
The names should be converted to title case and lower case as specified.	Yes	Applied transformation logic using InitCap() and Lower() functions.
The ROLL NO should be validated and converted to a number.	Yes	Applied validation and conversion logic in the transformation step.
The data load process should abort if any critical validation fails.	Yes	Implemented validation checks and abort logic in the ETL process.













