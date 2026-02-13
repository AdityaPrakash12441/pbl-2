"""
Endangered Species Detection Information
Shows what endangered species might be detected based on COCO classes
"""

print("=" * 70)
print("ENDANGERED SPECIES DETECTION GUIDE")
print("=" * 70)
print("\nYOLOv8n COCO Model Limitations:")
print("- Trained on 80 common objects, NOT specific endangered species")
print("- Can detect general categories that MAY include endangered animals")
print("\n" + "=" * 70)

DETECTION_GUIDE = {
    "Cat (Class 15)": {
        "description": "Detects felines - may identify big cats",
        "endangered_species": [
            "🔴 CRITICALLY ENDANGERED:",
            "  - Amur Leopard",
            "  - South China Tiger", 
            "  - Sunda Tiger",
            "  - Sumatran Tiger",
            "",
            "🟠 ENDANGERED:",
            "  - Bengal Tiger",
            "  - Indochinese Tiger",
            "  - Malayan Tiger",
            "  - Snow Leopard",
            "  - Clouded Leopard",
            "  - Jaguar",
            "  - Cheetah",
            "  - Asiatic Lion",
            "  - Iberian Lynx",
            "  - Fishing Cat",
            "  - Ocelot",
            "  - Margay"
        ]
    },
    "Dog (Class 16)": {
        "description": "Detects canines - may identify wild dogs/wolves",
        "endangered_species": [
            "🟠 ENDANGERED:",
            "  - African Wild Dog",
            "  - Ethiopian Wolf",
            "  - Dhole (Asian Wild Dog)",
            "  - Red Wolf"
        ]
    },
    "Elephant (Class 20)": {
        "description": "Detects elephants",
        "endangered_species": [
            "🔴 CRITICALLY ENDANGERED:",
            "  - Sumatran Elephant",
            "",
            "🟠 ENDANGERED:",
            "  - Asian Elephant",
            "  - African Forest Elephant"
        ]
    },
    "Bear (Class 21)": {
        "description": "Detects bears",
        "endangered_species": [
            "🟠 ENDANGERED/VULNERABLE:",
            "  - Polar Bear",
            "  - Giant Panda",
            "  - Sloth Bear",
            "  - Sun Bear",
            "  - Spectacled Bear",
            "  - Some Grizzly populations"
        ]
    },
    "Horse (Class 17)": {
        "description": "Detects equines - horses, zebras",
        "endangered_species": [
            "🟠 ENDANGERED:",
            "  - Przewalski's Horse",
            "  - Grevy's Zebra",
            "  - Mountain Zebra"
        ]
    },
    "Cow (Class 19)": {
        "description": "Detects bovines - cattle, buffalo, antelope",
        "endangered_species": [
            "🔴 CRITICALLY ENDANGERED:",
            "  - Addax",
            "  - Saiga Antelope",
            "  - Hirola",
            "",
            "🟠 ENDANGERED:",
            "  - Gaur",
            "  - Banteng",
            "  - Wild Yak",
            "  - European Bison",
            "  - Dama Gazelle"
        ]
    },
    "Sheep (Class 18)": {
        "description": "Detects sheep/goats",
        "endangered_species": [
            "🟠 ENDANGERED:",
            "  - Markhor",
            "  - Various wild sheep species"
        ]
    },
    "Giraffe (Class 23)": {
        "description": "Detects giraffes",
        "endangered_species": [
            "🟠 VULNERABLE/ENDANGERED:",
            "  - Various giraffe subspecies"
        ]
    },
    "Zebra (Class 22)": {
        "description": "Detects zebras",
        "endangered_species": [
            "🟠 ENDANGERED:",
            "  - Grevy's Zebra",
            "  - Mountain Zebra"
        ]
    },
    "Bird (Class 14)": {
        "description": "Detects birds - generic, not species-specific",
        "endangered_species": [
            "🔴 CRITICALLY ENDANGERED:",
            "  - Kakapo",
            "  - Philippine Eagle",
            "  - California Condor",
            "  - Spix's Macaw",
            "",
            "🟠 ENDANGERED:",
            "  - Whooping Crane",
            "  - Various Albatross species",
            "  - Various Vulture species",
            "  - Various Crane species",
            "  - Various Hornbill species",
            "  - Various Parrot species",
            "  - African Penguin",
            "  - Galápagos Penguin"
        ]
    }
}

for category, info in DETECTION_GUIDE.items():
    print(f"\n{category}")
    print("-" * 70)
    print(f"Detection: {info['description']}")
    print("\nPossible Endangered Species:")
    for species in info['endangered_species']:
        print(f"  {species}")

print("\n" + "=" * 70)
print("SPECIES NOT DETECTABLE (No COCO Class):")
print("=" * 70)
print("""
The following endangered species CANNOT be detected with YOLOv8n:
  - Javan Rhino
  - Sumatran Rhino
  - Black Rhino
  - Vaquita (porpoise)
  - Saola (Asian unicorn)
  - Gorillas (all species)
  - Orangutans (all species)
  - Chimpanzees, Bonobos
  - Gibbons, Lemurs
  - Most primates
  - Sea Turtles
  - Marine mammals (whales, dolphins, seals, manatees, dugongs)
  - Reptiles (tortoises, alligators, crocodiles, Komodo dragon)
  - Pangolins
  - Okapi
  - Takin
  - Red Panda
  - Tapirs
  - Most marsupials (Tasmanian Devil, Quokka, Tree Kangaroo)

To detect these, you would need a custom-trained model with wildlife datasets.
""")

print("=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("""
For accurate endangered species detection, consider:
1. Using specialized wildlife detection models
2. Fine-tuning YOLOv8 on endangered species datasets
3. Using models trained on camera trap data
4. Integrating with conservation databases like iNaturalist
""")
print("=" * 70)
