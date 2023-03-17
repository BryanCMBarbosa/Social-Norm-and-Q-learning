build:
	g++ -O2 src/Agent.cpp src/QAgent.cpp src/SocialAgent.cpp src/Simulation.cpp main.cpp -o simulation -std=c++17

build_p:
	g++-12 -O2 src/Agent.cpp src/QAgent.cpp src/SocialAgent.cpp src/Simulation.cpp main.cpp -o simulation -fopenmp -std=c++17

clear:
	@echo "Cleaning..."
	rm -f simulation
	@echo "Done."

clear_csv:
	rm -f *.csv

bash:
	python3 bash_file_generator.py 20 120 100000 50

clear_bash:
	rm -f *.sh

exec_all:
	./simulation 0000000000000000 All-bad 50
	./simulation 1111111111111111 All-good 50
	./simulation 1000100010001000 Shunning 50
	./simulation 1001100010011000 Judging 50
	./simulation 1001100110011001 Stern-judging 50
	./simulation 1001101010011010 Score-judging 50
	./simulation 1001101110011011 SJ+SS 50
	./simulation 1010101010101010 Image-score 50
	./simulation 1011100010111000 Strict-standing 50
	./simulation 1011100110111001 SS+SJ 50
	./simulation 1011101010111010 Standing 50
	./simulation 1011101110111011 Simple-standing 50

