from datasets.manager import DATASET_MENU
from geometry.calculator import GEOMETRY_MENU
from visualization.plots import VISUALIZATION_MENU


"""UAV-AERO ENGINE entry point."""


def sub_menu(menu_title, menu_dict):
    while True:
        print("\n" + "="*35)
        print(f"   {menu_title.upper()}   ")
        print("="*35)

        for key, item in menu_dict.items():
            print(f" {key}. {item["title"]}")

        print("[0] ⬅️ Back to Main Menu")
        print("="*35)

        choice = input(f"\n Enter your choice: ")

        if choice == "0":
            print("\n🔙 We are going to main menu.")
            break
        if choice in menu_dict:
            print("\n||||||||||||||||||||||||||||||||||||")
            menu_dict[choice]["func"]()
            print("||||||||||||||||||||||||||||||||||||")

            # PAUSE
            input(f"Press ENTER to return to menu.")


def main_menu():
    main_options = {
        "1" : {"title" : "Geometry Calculator", "menu": GEOMETRY_MENU},
        "2" : {"title" : "Dataset Manager", "menu" : DATASET_MENU},
        "3" : {"title" : " Graph Visualization", "menu" : VISUALIZATION_MENU},
    }

    while True:
        print("\n" + "★ "*35)
        print(" "*20 ,"WHAT WOULD YOU LIKE TO DO?")
        print("★ "*35)
        print("[1] Geometry Calculator")
        print("[2] Data Set Manager")
        print("[3] Plot Visualization")
        print("[0] Exit System")
        print("★ "*35)

        choice = input(f"\nEnter choice (0-3): ").strip()

        if choice == "0":
            print(f"Program is Ending. Good bye! 👋 ")
            break

        if choice in main_options:
            selected_title = main_options[choice]["title"]
            selected_dict = main_options[choice]["menu"]

            sub_menu(selected_title, selected_dict)
        else:
            print("\n❌ Wrong option! Please select 0, 1, 2 or 3.")



if __name__ == "__main__":
    main_menu()