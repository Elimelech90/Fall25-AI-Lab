tasks = []


def addTask():
    task = input("Enter a task: ")
    tasks.append(task)
    print(f"Task has been '{task}'added to the list")
    

def listTasks():
    if len(tasks) == 0:
        print("There are no tasks currently.")
    else:
        print("Current no of Tasks: ")
        for index, task in enumerate (tasks, start= 1):
            print(f"Task {index}. {task}")

    
def deleteTask():
    listTasks()
    try:
        taskToDelete = int(input("Enter the task number to delete: "))
        if taskToDelete >=0 and taskToDelete < len(tasks):
            tasks.pop(taskToDelete)
            print(f"Task number {taskToDelete} has been Deleted")
        else:    
            print(f"Task number {taskToDelete} was not found")
        
        
    except:
        print("Invalid input")
        
        
        
if __name__ == '__main__':
    #Create a loop to run the program 
    print("Welcome to the To Do List ")
    while True:
        print("\n")
        print("Please select one of the Following options: ")
        print("-----------------------------------------------")
        print("1. Add a new task")
        print("2. Delete a task")
        print("3. List tasks")
        print("4. Quit")
        
        choice= input("Enter your choice: ")
        
        if choice == "1":
            addTask()
        
        elif choice == "2":
            deleteTask()
    
        elif choice == "3":
            listTasks()
            
        elif choice == "4":
            break
        else:
            print("Invalid input. Please try again")

print("Program closed")
