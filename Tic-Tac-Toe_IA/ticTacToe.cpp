#include <iostream>
#include <GL/glut.h>
#include <string>
#include <utility>
#include "SOIL.h"

#include "ticTacToe.h"
#include "board.h"
#include <string>
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

typedef pair<int, int> pii;

Board board (3, 3);
int view_left, view_bottom, view_right, view_top;
int current_player = 1; ///Dado que siempre empieza 'X'
int state = 1; 
unsigned int crossId = 1; ///ID de 'X'
unsigned int oId = 2;     ///ID de 'O'
bool againstComputer = false;

int level = 6;

void display(void)
{
    fill_board();
    draw_initial_board();
    
    int winner = check_win();
    
    if (winner != 0) {
        display_win(winner);
    }
    else if (find_valid_moves(current_player).size() == 0) {
        display_win(0);
    }
    
    glutSwapBuffers();
}

int main(int argc, char *argv[])
{
	 
    if (argc == 2 ) {
       level = atoi(argv[1]);
    }
    
    
    againstComputer = true;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Tic Tac Toe");

    init();
    
    glutMouseFunc(process_mouse);
    glutDisplayFunc(display);
    glutMainLoop();

    return 0;
}

void process_mouse(int button, int buttonState, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && buttonState == GLUT_UP) {
        if (state == 1) {
            int i = (y / 500.0 * 100 - 5) / 30;
            int j = (x / 500.0 * 100 - 5) / 30;
            if (inside(i, j) && board.getXY(i, j) == 0) {
                move_player(i, j);
            }
        } else {
            reset_board();
        }
    }
}

void move_player(int i, int j)
{
    board.setXY(i, j, current_player);
    display();
    if (check_win() != current_player) { ///Siempre en la jugada actual, se comprueba si ha surgido ganador tras la movida
        current_player = 3 - current_player;
        if (againstComputer) {
            move_computer();
        }
    }
}

void move_computer()
{
    pii next_move = find_next_move(current_player, level).second;
    if (next_move.first == -1) {
        return;
    }
    board.setXY(next_move.first, next_move.second, current_player);
    current_player = 3 - current_player;
    display();
}

void reset_board(void)
{
    state = 1;
    current_player = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            board.setXY(i, j, 0);
        }
    }
    glColor3f(0.06, 0.06, 0.06);
    glBegin(GL_POLYGON);
        glVertex2f(view_left, view_bottom);
        glVertex2f(view_right, view_bottom);
        glVertex2f(view_right, view_top);
        glVertex2f(view_left, view_top);
    glEnd();

    display();
}

pair< int, pii> find_next_move(int player, int max_player)
{
    vector<pii> valid_moves = find_valid_moves(player);
    if (check_win() != 0 || valid_moves.size() == 0) {
        return terminal_utility(max_player);
    }
    int bestVal;
    pii bestMove (-1, -1);
    if (player == max_player) {
        bestVal = -2;
    } else {
        bestVal = 2;
    }
    ///cout<<"bestVal:"<<bestVal<<", player: "<<player<<", max_player:"<<max_player<<endl;
    for (int i = 0; i < valid_moves.size(); i++) {
        board.setXY(valid_moves[i].first, valid_moves[i].second, player);
        int currVal = find_next_move(3 - player, max_player).first;
        if (player == max_player) {
            if (currVal > bestVal) {
                bestVal = currVal;
                bestMove = valid_moves[i];
            }
        } else  if (currVal < bestVal) {
                bestVal = currVal;
                bestMove = valid_moves[i];
        }
        
        board.setXY(valid_moves[i].first, valid_moves[i].second, 0);
    }
    
    return make_pair(bestVal, bestMove);
}

vector<pii> find_valid_moves(int player)
{
    vector<pii> valid_moves;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board.getXY(i, j) == 0) {
                valid_moves.push_back(make_pair(i, j));
            }
        }
    }
    return valid_moves;
}

int check_win(void)
{
    for (int i = 0; i < 3; i++) {
        if (check_row(i) != 0) {
            return check_row(i);
        }
        if (check_col(i) != 0) {
            return check_col(i);
        }
    }
    if (board.getXY(0, 0) != 0 && board.getXY(0, 0) == board.getXY(1, 1) && board.getXY(0, 0) == board.getXY(2, 2)) {
        return board.getXY(0, 0);
    }
    if (board.getXY(0, 2) != 0 && board.getXY(0, 2) == board.getXY(1, 1) && board.getXY(0, 2) == board.getXY(2, 0)) {
        return board.getXY(0, 2);
    }
    return 0;
}

pair<int, pii> terminal_utility(int max_player)
{
    int winner = check_win();
    if (winner == max_player) {
        return make_pair(1, make_pair(-1, -1));
    } else if (winner == 3 - max_player) {
        return make_pair(-1, make_pair(-1, -1));
    } else {
        return make_pair(0, make_pair(-1, -1));
    }
}

void display_win(int winner) 
{
    glColor3f(1.0, 1.0, 1.0);

    std::string message = "Jugador 0 gana!";
    message[7] = winner + '0';
    if (winner == 0) {
        message = "Empate!";
    }

    glRasterPos2f(40, 50);
    print_message(message);
    
    glColor3f(1.0, 0.0, 0.0);
    std::string click_msg = "Click en cualquier parte para reiniciar";
    glRasterPos2f(27, 5);
    print_message(click_msg);

    state++;
}

bool loadTexture(char *filename, unsigned int *texture){
    *texture = SOIL_load_OGL_texture(filename,
            SOIL_LOAD_AUTO,
            SOIL_CREATE_NEW_ID,
            SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_MULTIPLY_ALPHA
            );
    if(*texture == 0){
        return false;
    } else {
        return true;
    }
}

int check_row(int x) 
{
    if (board.getXY(x, 0) == 0) {
        return 0;
    }
    for (int i = 1; i < 3; i++) {
        if (board.getXY(x, i) != board.getXY(x, 0)) {
            return 0;
        }
    }
    return board.getXY(x, 0);
}

int check_col(int y)
{
    if (board.getXY(0, y) == 0) {
        return 0;
    }
    for (int i = 1; i < 3; i++) {
        if (board.getXY(i, y) != board.getXY(0, y)) {
            return 0;
        }
    }
    return board.getXY(0, y);
}

void init(void)
{
    glClearColor(0.06, 0.06, 0.06, 0.0);

    glMatrixMode(GL_MODELVIEW);
    
    view_left = 0;
    view_right = 100;
    view_bottom = 0;
    view_top = 100;

    gluOrtho2D(view_left, view_right, view_bottom, view_top);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!loadTexture("textures/x.jpg", &crossId)) {
        cout << "Unable to load cross" << endl;
    }
    if (!loadTexture("textures/o.jpg", &oId)) {
        cout << "Unable to load o" << endl;
    }
}

void fill_board(void)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board.getXY(i, j) == 0) {
                continue;
            } else {
                fill_player(j, i, board.getXY(i, j));
            }
        }
    }
}

void fill_player(int x, int y, int player) 
{
    unsigned int textureId;

    if (player == 1) {
        textureId = crossId;
    } else {
        textureId = oId;
    }
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glBegin(GL_POLYGON);
        glTexCoord2f(0, 0);
        glVertex2f(x * 30 + 5, (3 - y - 1) * 30 + 5);

        glTexCoord2f(1, 0);
        glVertex2f(x * 30 + 5 + 30, (3 - y - 1) * 30 + 5);

        glTexCoord2f(1, 1);
        glVertex2f(x * 30 + 5 + 30, (3 - y -1) * 30 + 5 + 30);

        glTexCoord2f(0, 1);
        glVertex2f(x * 30 + 5, (3 - y - 1)* 30 + 5 + 30);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void draw_initial_board(void)
{
    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINES);
        for (int i = 1; i < 3; i++) {
            glVertex2f(view_left + 5, i * 30 + 5);
            glVertex2f(view_right - 5, i * 30 + 5);

            glVertex2f(i * 30 + 5, view_bottom + 5);
            glVertex2f(i * 30 + 5, view_top - 5);
        }

    glEnd();
}

void print_message(std::string message)
{
    for (int i = 0; i < message.size(); i++) {
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, message[i]);
    }
}

bool inside(int x, int y)
{
    if (x < 0 || x >= 3) {
        return false;
    }
    if (y < 0 || y >= 3) {
        return false;
    }
    return true;
}

