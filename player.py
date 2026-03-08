
import chess
import pandas as pd
from minicons import scorer
import chess
import torch
from chess_tournament import Player
    

device = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerPlayer(Player):
    def __init__(self, name, hfId = "akshaydwj/chess-distilgpt2", depth=4):
        super().__init__(name)
        self.hfId = scorer.IncrementalLMScorer(hfId, device=device)
        self.depth = depth # depth checks how far the model goes for thinking ahead

    def prompting(self, fen, move):
        # this funciton gives the correct notation so that the model can correctly asses what needs to be played
        return f"FEN: {fen} MOVE: {move}"

    def positionChecking(self, fen):
        # this checks the current position and how we are standing for it 
        board = chess.Board(fen)
        legal_moves = []
        for x in board.legal_moves:
            legal_moves.append(x.uci())
        if not legal_moves:
            return -999 if board.is_checkmate() else 0

        restricet = (30 - len(legal_moves)) * 0.1 # this restricts it so we 
        prompts = []
        for x in legal_moves[:4]:
            prompts.append(self.prompting(fen,x))
        scores = self.hfId.sequence_score(prompts)
        return max(scores) + restricet

    def minimax(self, board, depth , is_maximizing, alpha=-9999, beta=9999): #minimaxing is a tactic in chess that you can use to make it so that if opponents is optimal it minimizes the win it gets on the board
        #thi is the only function that could put some withstanding against the stockfish strong even though sometimes mates do get missed because of it
        if board.is_checkmate():
            return -9999 if is_maximizing else 9999
        if board.is_stalemate() or board.is_repetition():
            return 0
        if depth == 0:
            return self.positionChecking(board.fen())
        legal_moves = list(board.legal_moves)
        if is_maximizing:
            #looks for the mest move in minimizing
            best = -9999
            for move in legal_moves[:4]:

                board.push(move)
                score = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                best = max(best, score)

                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
                        #looks for the mest move in maximizing
            best = 9999
            for move in legal_moves[:4]:
                board.push(move)
                score = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()

                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:

                    break

            return best

    def get_move(self, fen):
        # this function checks what move is legal and what is the best current move to play
        board = chess.Board(fen)
        legalMoves = list(board.legal_moves)
        if not legalMoves:
            return None


        bestScore = -9999
        best_move = None
        alpha = -9999



        for move in legalMoves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            
            score = self.minimax(board, self.depth - 1, False, alpha, 9999)
            board.pop()


            if score > bestScore:
                bestScore = score
                best_move = move
                alpha = max(alpha, bestScore)

        return best_move.uci() if best_move else None
