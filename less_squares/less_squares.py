# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:24:26 2024

@author: cdarc

Notes
-----
The code should have the following structure

A should be stored as a skinny OR square matrix, with a "transposed" flag for 
if it is actually fat.

Furthermore it should be stored with its collumns normalized. This way 
computation is more stable and the true matrix can be recreated later.

We only care about:
    -append
    -delete
    -add

External operations for the user use the 'intended' axis which is based on the 
input matrix. Internal operations use the axis as applied to the stored skinny 
matrix.
"""

import numpy as np
import warnings

class LessSquares:
    def __init__(self, matrix):
        """
        

        Parameters
        ----------
        matrix : (M, N) array_like
            input matrix from which pseudoinverse is calculated.

        Returns
        -------
        None.

        """
        self.is_transposed = matrix.shape[1]>matrix.shape[0]
        if self.is_transposed:
            self.norms = np.linalg.norm(matrix,axis=1)
            self.A = (np.array(matrix).T/self.norms)
        else:
            self.norms = np.linalg.norm(matrix,axis=0)
            self.A = (np.array(matrix)/self.norms)       
        self.pinv = np.linalg.pinv(self.A,rcond=1e-35)
        self.op_counter = 0
        self.err_tol = 1e-10
        self.check_period = 50
        self.checking = False
    
    @property
    def pseudo(self):
        if self.is_transposed:
            return self.pinv.T/self.norms
        else:
            return self.pinv/self.norms[:,np.newaxis]
    @property
    def matrix(self):
        if self.is_transposed:
            return self.A.T*self.norms[:,np.newaxis]
        else:
            return self.A*self.norms
        
    def _internal_transpose(self):
        self.is_transposed = not self.is_transposed
        self.A = (self.A * self.norms).T
        self.pinv = (self.pinv/self.norms[:,np.newaxis]).T
        self.norms = np.linalg.norm(self.A,axis=1)
        self.A = (self.A/self.norms)
        self.pinv = self.pinv*self.norms[:,np.newaxis]   
    
    def _op_update(self):
        self.op_counter += 1
        if self.checking == True and self.op_counter % self.check_period:
            if self.check('fast'):
                warnings.warn('check-failed')
                #self.pinv = np.linalg.pinv(self.A)
    
    def _validate_index(self, index, intended_axis, axis):
        """Helper function to validate index based on the matrix shape and axis.
        robust to all non-functional index selections
        """
        try:
            corrected_index = int(index)
        except:
            raise TypeError(f"Non-integer index {str(index)} provided.")
        if not -self.A.shape[intended_axis] <= corrected_index < self.A.shape[intended_axis]:
            selector = (axis)%2
            if selector == 0:
                raise IndexError(f"Row index {corrected_index} is out of bounds for matrix with {self.A.shape[intended_axis]} rows.")
            elif selector == 1:  # Column check
                raise IndexError(f"Column index {corrected_index} is out of bounds for matrix with {self.A.shape[intended_axis]} columns.")
        else:
            return corrected_index
    
    def _validate_vector(self, vector, intended_axis, axis):
        try:
            vector = vector.flatten()
        except:
            raise TypeError("vector must be either a numpy array or a list.")
        if len(vector) != self.A.shape[(intended_axis+1)%2]:
            raise ValueError(f"Dimension mismatch, length {len(vector)} input for {((axis)%2)*'row' + ((1+axis)%2)*'column'} operation does not match matrix {((axis)%2)*'row' + ((1+axis)%2)*'column'} of length {self.A.shape[intended_axis]}.")
        elif vector.dtype.name not in ['float64','float','bool','int','int_']:
            raise ValueError(f"Invalid data type. Vector data type must be one of float64, float, bool, int, or int_. Input vector has data type {vector.dtype.name}")
        else:
            return vector
    
    def _type_validator(self,inputs):
        keys = inputs.keys()
        for k in keys:
            if 'axis' in k:
                if type(inputs[k]).__name__ == 'ndarray':
                    if len(inputs[k].shape)==2:
                        if inputs[k].dtype.name not in ['float64','float','bool','int','int_']:
                            ValueError(f"Invalid data type. Matrix must be one of float64, float, bool, int, or int_. Input matrix has data type {inputs[k].dtype.name}")
                    else:
                        ValueError(f"Invalid shape. Matrix must have two dimensions, input matrix has {len(inputs[k].shape)}")
                else:
                    raise TypeError("Not a numpy array")
            elif 'index' in k:
                raise TypeError("do later")
            elif 'matrix' in k:
                raise TypeError("do later")
    
    def _intended_axis(self,axis):
        """
        Robust to all non-functional axis selections
        """
        try:
            if self.is_transposed:
                intended_axis = (axis+1)%2
            else:
                intended_axis = (axis)%2
        except:
            raise TypeError("Invalid axis type. Axis must be either 0, 1, or their boolean/float equivalents.")
        if intended_axis in [0,1]:
            return intended_axis
        else:
            raise TypeError("Invalid axis type. Axis must be either 0, 1, or their boolean/float equivalents.")
   
    def expand(self, axis):
        """
        Adds a "junk" row or collumn to the matrix. That can be later updated.
        Notes:
            -use the last column of A to do underdetermined updates, not
               the first, as this way we can do iterative additions easier
            -breaks if you double expand
        
        Parameters
        ----------
        axis : {0,1}
            Specifies the axis of `self.matrix` along which to
            expand the matrix. If axis is 0, it will append a row to the 
            bottom, if axis is 1, it will append a collumn to the side.

        Returns
        -------
        None.

        """
        intended_axis = self._intended_axis(axis)
        if self.A.shape[0] == self.A.shape[1]:
            if intended_axis == 1:
                self._internal_transpose()
                intended_axis = 0
        
        if intended_axis == 0:
            zn = np.zeros(shape=(1,self.A.shape[1]))
            self.A = np.vstack((self.A,zn))
            self.pinv = np.hstack((self.pinv,zn.T))  
        else:
            def nuller(n,x0=[]):
                if np.array_equal(x0,[]):
                    x0 = self.A @ np.sum(self.pinv,axis=1)
                    x0 -= 1
                else:
                    x0 -= self.A @ (self.pinv @ x0)
                if n>1:
                    return nuller(n-1,x0)
                else:
                    return x0/np.linalg.norm(x0)
            yn = nuller(max(25-self.A.shape[0]+self.A.shape[1],2))
            zn = yn.flatten()
            self.A = np.hstack((self.A,zn[:,np.newaxis]))
            self.pinv = np.vstack((self.pinv,yn))
            self.norms = np.append(self.norms,1)
    
    def _expander_conditioner(self,A,P,h,z):
        """
        Needs more umph

        """
        Pf = np.vstack((P,z[np.newaxis,:]))
        aph = (A@(P@h))
        zap = ((z @ A) @ P)
        E = np.vstack((np.outer(P@h,z),zap))
        dP = (Pf-E)
        p1 = np.outer(dP@aph,z)
        p1 -= np.outer(dP@z,aph)
        p2 = np.outer(dP@h,zap)
        p2 -= np.outer(dP@zap,h)
        tau = np.outer(dP@z,h) - np.outer(dP@h,z)
        Af = np.hstack((A,h[:,np.newaxis]))
        return Af,dP

    def append(self, vector, axis):
        """
        Append a vector to the matrix along the chosen axis. 
        0 will add a row to the bottom, 1 will add a collumn to the side.

        Parameters
        ----------
        vector : {(M,), (M, 1), (1, M)} array_like
            Vector to be appended to the matrix. If `vector` is two-dimensional,
            it will be flattened. Does not support matrix appending.
        axis : {0,1}
            Specifies the axis of `self.matrix` along which to
            append the vector. If axis is 0, it will append a row to the 
            bottom, if axis is 1, it will append a collumn to the side.

        Returns
        -------
        None.

        """
        self.expand(axis)
        intended_axis = self._intended_axis(axis)
        vector = self._validate_vector(vector, intended_axis, axis)
        if intended_axis == 0:
            rescaling = np.sqrt(1 + ((vector/(self.norms)).flatten())**2)
            self.norms *= rescaling
            vector = vector/(self.norms)
            rescaling = rescaling
            if np.any(np.abs(rescaling)>10000):
                warnings.warn('The values in this update will dominate several collumns, resulting in loss of precision. Even stock solvers like numpy will fail to solve this well and will fail the checks. Feel free to examine the results of this rescaled matrix with numpy applied. We should maybe create a safe mode that rejects inputs like this gracefully. Auto-outlier or something. You can show this to yourself by checking that z increases as k increases for A = np.random.normal(size=(10,10))*np.exp(k*np.random.normal(size=10)); z = np.linalg.norm(np.eye(10)-np.linalg.pinv(A) @ A)')
            self.A = self.A/(rescaling[np.newaxis,:])
            self.pinv = self.pinv*rescaling[:,np.newaxis]
            self._blank_update(vector,-1)
        else:
            current_val = self.A[:,-1]
            updated_col = vector-(current_val*self.norms[-1])
            self.add(updated_col,-1,axis,_append_mode=True)
    
    def add(self,vector,index,axis,_append_mode=False):
        """
        Add a specified vector to the matrix at a specific index.
        Setting the axis to 0 will add to the row specified by the index.
        Setting the axis to 1 will add to the column specified by the index.
        
        Notes: speed up the long last case, it is most common and least optimized.        

        Parameters
        ----------
        vector : {(M,), (M, 1), (1, M)} array_like
            Vector to be added to the matrix. If `vector` is two-dimensional,
            it will be flattened. Does not support matrix appending.
        index : int
            Index along the specified axis where the vector will be added.
        axis : {0,1}
            Specifies the axis of `self.matrix` along which to
            append the vector. If axis is 0, it will append a row to the 
            bottom, if axis is 1, it will append a collumn to the side.
        _append_mode : bool, optional
            Internal flag for use if we are adding to a recently expanded 
            matrix. The default is False.


        Returns
        -------
        None.

        """
        #input correction
        
        intended_axis = self._intended_axis(axis)
        
        if not _append_mode:
            vector = self._validate_vector(vector, intended_axis, axis)
            index = self._validate_index(index, intended_axis, axis)
        
        #This variable exists for readability
        collumn_update = ((intended_axis+1)%2 == 0)
        
        #norm and vector updates
        if collumn_update:
            Mp = vector+self.norms[index]*self.A[:,index]
            Normp = np.linalg.norm(Mp)
            u = (Mp/Normp) - self.A[:,index]
            self.norms[index] = Normp
        else:
            normp = np.sqrt(self.norms*(2*self.A[index,:]*vector+self.norms)+(vector**2))
            self.A *= self.norms/normp
            self.pinv *= (normp/self.norms)[:,np.newaxis]
            self.norms = normp
            
            v = vector/self.norms
        
        if collumn_update:#v is indexer
            if self.A.shape[0] == self.A.shape[1]:
                gamma = self.pinv[index,:]
                oputg = 1 + np.inner(u,gamma)
                self.pinv -= np.outer(self.pinv@u,gamma/oputg)
                self.A[:,index] += u
            else:
                self._full_update(u,index)
        else:
            gamma = (v @ self.pinv)
            utg = gamma[index]
            if self.A.shape[0] == self.A.shape[1]:
                #u logic
                self.pinv -= np.outer(self.pinv[:,index],(gamma/(1+utg)))
            else:
                #u logic
                #This is the most common case, we need to speed this up
                utA = self.A[index,:]
                utAA_p = utA @ self.pinv
                utAA_pu = utAA_p[index]
                gtg = np.inner(gamma,gamma)
                oputg = 1+utg
                Zinv = np.array([[oputg,-utg-utAA_pu],[-gtg,oputg+gtg]])
                Zdet = (Zinv[0,0]*Zinv[1,1])-(Zinv[1,0]*Zinv[0,1])
                if np.abs(Zdet) < 0.0000000000001:
                    warnings.warn('Unexpected singularity in data, inform developers')
                    tz = (self.pinv.T@self.pinv[:,index])
                    tz = -tz/tz[index]
                    if self.A.shape[1]<=self.A.shape[0]:
                        self.pinv = (self.pinv + np.outer(self.pinv[:,index],tz))
                    else:
                        self.pinv = (self.pinv + np.outer(self.pinv[:,index],tz)).T
                else:
                    A_pg = self.pinv@gamma
                    tau = self.pinv[:,index]
                    guess = np.outer(A_pg,utAA_p*(oputg)+gamma*(1-utAA_pu))
                    guess += np.outer(tau,(oputg*gamma-utAA_p*gtg))
                    guess[:,index] += tau*gtg - A_pg*oputg
                    guess = guess/Zdet
                    self.pinv -= guess
            self.A[index,:] += v
        self._op_update()
          
    def _local_update(self,v_local,u,index):
        """
        Future updates:
            print('u',u[index])
            print('del',np.linalg.norm(A_po[index,:]-Ao[:,index]))
        in flat situations when u is zero we can expect A_po[index] == Ao[index],
        allowing us to exploit 0 error in Ao.
        """
        self.A[:,index] += v_local
        gamma = self.pinv[index,:]
        u_lower = u[np.delete(np.arange(u.shape[0]), index)].copy()
        if np.abs(u[index]+1)<0.0000001:
            scalar = 1/(np.linalg.norm(u)**2)
            #https://www.youtube.com/watch?v=PKL2uB1gNK0
            rp = ((u @ self.pinv)-u[index]*gamma)*scalar
            guess = np.outer(-u,rp)
            guess[index,:] -= gamma
            self.pinv += guess
        else:
            scalar = (1+u[index])
            lower_guess = np.outer(-u_lower/scalar,gamma)
            upper_guess = -(u[index]/scalar)*gamma
            self.pinv[index,:] += upper_guess
            self.pinv[np.delete(np.arange(self.pinv.shape[0]), index),:] += lower_guess

    def _non_local_update(self,u,index):
        #should it be -u or gamma in that term?
        gamma = self.pinv[index,:]
        gtg = np.inner(gamma,gamma)
        utu = np.inner(u,u)
        utA = u @ self.A
        utAA_p = utA @ self.pinv
        utAA_pu = np.inner(utAA_p,u)
        A_pu = self.pinv@u
        if np.isclose(gtg,1):
            gtg = 1;
            utg = utA[index];
            oputg = utg+1
            Zdet = oputg**2 + utu - utAA_pu
            R = (-u-(oputg*gamma) + utAA_p)/Zdet
            self.pinv[index,:] -= gamma+(oputg*R)
            self.pinv += np.outer(A_pu,R)
            self.A[:,index] += u
        elif np.isclose(gtg,0):
            warnings.warn('LessSquares object is being operated on with null pseudoinverse entries. Inform developers of what you did to trigger this.')
            self.A[:,index] += u
        else:
            A_pg = self.pinv@gamma
            utg = np.inner(u,gamma)
            oputg = utg+1
            Zinv = np.array([[oputg,-(utu*utg)-utAA_pu],[-gtg,oputg+gtg*utu]])
            Zdet = (Zinv[0,0]*Zinv[1,1])-(Zinv[1,0]*Zinv[0,1])
            guess = np.outer(A_pg,-gamma/gtg)
            guess += np.outer( gtg*A_pu - oputg*A_pg, (-u-(oputg/gtg)*gamma + utAA_p)/Zdet)
            self.pinv += guess
            self.A[:,index] += u

    def _full_update(self,v,index = -1):
        c = self.pinv @ v
        v_local = self.A @ c
        v_nonlocal = v-v_local
        self._local_update(v_local,c,index)
        self._non_local_update(v_nonlocal,index)

    def _blank_update(self,v,index):
        """A special case of the add operation"""
        self.A[index,:] += v.flatten()
        #u logic
        gamma = (v.T @ self.pinv).T
        gtg = (gamma.T @ gamma).item()
        fr = gamma/(1+gtg)
        rb = gamma.flatten()
        special_k = self.pinv@fr
        self.pinv -= np.outer(special_k,rb)
        self.pinv[:,index] += special_k.flatten()
        self._op_update()

    def _slice_zero(self,index,axis):
        """Set a row or collumn to zero. The pinv doesn't change, neat huh?"""
        collumn_zero = ((axis+1)%2 == 0)
        if collumn_zero:
            if np.any(self.A[:,index]):
                self.norms[index] = 0
                tz = (self.pinv[index,:]@self.pinv.T)
                tz = -tz/tz[index]
                self.pinv += np.outer(tz,self.pinv[index,:])
                self.A[:,index] = 0        
        else:
            if np.any(self.A[index,:]):
                old_norms = self.norms.copy()
                self.norms = np.sqrt(self.norms**2 - self.A[index,:].flatten()**2)
                rescaling = (old_norms[np.newaxis,:]/self.norms[np.newaxis,:])
                
                if self.A.shape[0] > self.A.shape[1]:
                    tz = (self.A @ self.pinv[:,index])
                    cst = 1-tz[index]#(self.A[index,:] @ self.pinv[:,index])
                    tz = -tz/cst
                    tz[index] = 1
                    self.pinv -= np.outer(self.pinv[:,index],tz)
                    self.A = self.A*rescaling
                    self.pinv = self.pinv/(rescaling.T)
                    self.A[index,:] = 0
                else:
                    tz = self.pinv.T @ self.pinv[:,index]
                    tz = -tz/tz[index]
                    self.pinv += np.outer(self.pinv[:,index],tz)
                    self.A[index,:] = 0
                    self.A = self.A/rescaling
    
    def delete(self,index,axis):
        """
        Delete a row or column from the model.

        Parameters
        ----------
        index : int
            Index along the specified axis that will be deleted.
        axis : {0,1}
            Specifies the axis of `self.matrix` along which a row/column will 
            be deleted. If axis is 0, it will delete a row at the specified 
            index, if axis is 1, it will delete a collumn at the specified 
            index.

        Returns
        -------
        None.

        """
        intended_axis = self._intended_axis(axis)
        index = self._validate_index(index, intended_axis, axis)
        
        if self.A.shape[0] == self.A.shape[1]:
            if intended_axis == 0:
                self._internal_transpose()
                intended_axis = 1
        
        axis = intended_axis
        
        #check if this causes a flip, then update norms
        self._slice_zero(index,intended_axis)
        if ((intended_axis+1)%2 == 0):
            self.norms = np.delete(self.norms,index,0)
            if ((index+1)%self.A.shape[1])==0:
                self.pinv = self.pinv[:-1,:]
                self.A = self.A[:,:-1]
            else:
                self.pinv = np.vstack((self.pinv[:index,:],self.pinv[index+1:,:]))
                self.A = np.hstack((self.A[:,:index],self.A[:,index+1:]))
        else:
            if ((index+1)%self.A.shape[0])==0:#end delete
                self.pinv = self.pinv[:,:-1]
                self.A = self.A[:-1,:]
            else:
                self.pinv = np.hstack((self.pinv[:,:index],self.pinv[:,index+1:]))
                self.A = np.vstack((self.A[:index,:],self.A[index+1:,:]))
    
    def check(self,mode='full'):
        """
        checks the validity of an inverse/pseudoinverse pair. For 'fast' mode 
        it will return True/False for pass or fail while 'full' will return 
        the maximal pointwise error within the test cases.

        Parameters
        ----------
        mode : string, optional
            DESCRIPTION. The default is 'Full'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if mode=='full':
            A = self.A.copy()
            A_p = self.pinv.copy()
            c1 = A @ A_p @ A - A
            c2 = A_p @ A @ A_p - A_p
            c3 = (A @ A_p).T - (A @ A_p)
            c4 = (A_p @ A).T - A_p @ A
            return np.max(np.abs(c1)),np.max(np.abs(c2)),np.max(np.abs(c3)),np.max(np.abs(c4))
        else:
            c2 = np.max(((self.pinv @ self.A)-np.eye(self.A.shape[1])))
            return c2>self.err_tol
    
    def swap_slices(self, axis, index1, index2):
        """
        Swaps a pair of rows or collumns in the matrix
        
        Parameters
        ----------
        axis : {0,1}
            Specifies the axis of `self.matrix` along which rows/columns will 
            be swapped. If axis is 0, it will swap the rows at the specified 
            indices, if axis is 1, it will swap the collumns at the specified 
            indices.
        index1 : int
            Index of a row/column that will be swapped.
        index2 : int
            Index of a different row/column that will be swapped.

        """
        intended_axis = self._intended_axis(axis)
        index1 = self._validate_index(index1, intended_axis, axis)
        index2 = self._validate_index(index2, intended_axis, axis)
        def swap_slices_internal(input_matrix, axis, index1, index2):
            # Use tuple indexing directly to access and swap slices
            matrix = input_matrix.copy()
            a = matrix.take(index1, axis=axis).copy()
            b = matrix.take(index2, axis=axis).copy()

            # Assign the slices to their new positions
            matrix[(slice(None),) * axis + (index1,)] = b
            matrix[(slice(None),) * axis + (index2,)] = a
            return matrix
        self.A = swap_slices_internal(self.A, intended_axis, index1, index2)
        self.pinv = swap_slices_internal(self.pinv, ((intended_axis+1)%2), index1, index2)

    def update(self,vector,index,axis):
        """Not unit tested"""
        intended_axis = self._intended_axis(axis)
        index = self._validate_index(index, intended_axis, axis)
        vector = vector.flatten()
        if intended_axis == 1:
            update_vector = vector-(self.norms[index]*self.A[:,index,np.newaxis]).flatten()
        else:
            update_vector = vector-(self.norms*self.A[index,:,np.newaxis]).flatten()
        self.add(update_vector,index,axis)